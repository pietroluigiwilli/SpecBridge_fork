#!/usr/bin/env python3
"""
Predict SMILES for each spectrum in an MGF file using SpecBridge.

This script:
- Reads spectra from MGF file
- Uses FEATURE_ID to look up candidates from JSON
- Predicts the best SMILES for each spectrum
- Outputs results to a file

Handles duplicate identifiers by keeping the first occurrence (or can combine).
"""
from __future__ import annotations
import argparse
import json
import os
from typing import Dict, List, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from specbridge.utils.common import set_seed
from specbridge.adapters.dreams_adapter import load_dreams_encoder, DreamsAdapter
from specbridge.models.mol import MolFeaturizer, MolEncoder, MolAdapter
from specbridge.models.mapper import MapperB, DreamsToMolCondition
from specbridge.data.massspecgym import MassSpecGymDataset, collate_massspecgym


def _canon_smi(s: Optional[str]) -> Optional[str]:
    """Canonicalize SMILES if RDKit is available; else strip whitespace."""
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(mol) if mol is not None else s
    except Exception:
        return s


def build_mol_embed_fn(args, model, device):
    """
    Returns a function: embed_fn(List[str]) -> torch.FloatTensor [N, cond_dim] (on CPU)
    Uses the model's chem_proj to project embeddings into the same space as the query.
    """
    if args.mol_space == "chemberta":
        for p in model.parameters():
            p.requires_grad = False

        def embed_fn(smiles_list: list[str], device, bs: int = 512) -> torch.Tensor:
            # Match candidates.py: return raw ChemBERTa embeddings (not projected)
            # This ensures candidates are in the same space as z_m from the model
            if not smiles_list:
                # Return empty tensor with correct shape - use ChemBERTa hidden size
                dummy_h = model._chemberta_embed(["C"], device)
                out_dim = dummy_h.shape[-1]
                return torch.empty((0, out_dim), dtype=torch.float32)
            
            out = []
            with torch.no_grad():
                for i in range(0, len(smiles_list), bs):
                    chunk = smiles_list[i:i+bs]
                    if not chunk:  # Skip empty chunks
                        continue
                    # Get raw ChemBERTa embeddings (same as candidates.py line 153)
                    h = model._chemberta_embed(chunk, device)  # [B, hidden_size] = [B, 768]
                    out.append(h.cpu())
            
            if not out:
                # Fallback if all chunks were empty
                dummy_h = model._chemberta_embed(["C"], device)
                out_dim = dummy_h.shape[-1]
                return torch.empty((0, out_dim), dtype=torch.float32)
            
            Z = torch.cat(out, dim=0)
            return Z  # Raw ChemBERTa embeddings, not projected
        return embed_fn
    else:
        raise ValueError(f"Unknown --mol-space: {args.mol_space}")


def build_model(args, device):
    """Build and load the SpecBridge model."""
    dreams_backbone = load_dreams_encoder(args.dreams_ckpt, d_in=args.spec_bins, d_out=1024)

    model = DreamsToMolCondition(
        dreams_backbone,
        d_out=args.cond_dim,
        mapper_hidden=args.mapper_hidden,
        gaussian=not args.no_gaussian,
        mol_space=args.mol_space,
        chemberta_model=getattr(args, "chemberta_model", None),
        args=args
    ).to(device).eval()

    if args.adapter_ckpt:
        state = torch.load(args.adapter_ckpt, map_location="cpu")
        print(f"[model] Loading adapter checkpoint: {args.adapter_ckpt}")
        missing, unexpected = model.load_state_dict(state.get("model", {}), strict=False)
        if missing:
            print(f"[model] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[model] Unexpected keys: {len(unexpected)}")

    for p in model.parameters():
        p.requires_grad = False

    return model


class FeatureIDDataset(torch.utils.data.Dataset):
    """Dataset that preserves FEATURE_ID for lookup."""
    def __init__(self, mgf_path: str, normalize_intensities: bool = True):
        super().__init__()
        try:
            from pyteomics import mgf
        except Exception as e:
            raise ImportError(f"pyteomics is required: {e}")

        self.normalize_intensities = normalize_intensities
        self._records = []
        with mgf.MGF(mgf_path) as reader:
            for spec in reader:
                params = spec.get('params', {})
                feature_id = params.get('FEATURE_ID') or params.get('feature_id') or params.get('FEATUREID')
                title = params.get('NAME') or params.get('name') or params.get('Title') or params.get('TITLE') or f"idx_{len(self._records)}"
                
                # Store both feature_id and title for lookup
                rec = {
                    "title": str(title),
                    "feature_id": str(feature_id) if feature_id else None,
                    "mz": torch.tensor(spec.get('m/z array'), dtype=torch.float32),
                    "intensity": torch.tensor(spec.get('intensity array'), dtype=torch.float32),
                    "params": params,
                }
                self._records.append(rec)

    def __len__(self):
        return len(self._records)

    def __getitem__(self, idx: int):
        rec = self._records[idx]
        # Normalize intensities to [0, 1] by dividing by max intensity
        # This matches the expected format for DreaMS (intensities normalized in __getitem__)
        # Required for spectra with raw intensities (e.g., all_spectra.mgf)
        intensity = rec["intensity"].clone()
        if self.normalize_intensities:
            max_intensity = intensity.max()
            if max_intensity > 0:
                intensity = intensity / max_intensity
        
        # Create minimal meta for collate function
        # Note: smi_key is required by collate_massspecgym, but we use dummy SMILES since we're predicting
        meta = {
            "formula_idx": None,
            "adduct_idx": None,
            "charge_idx": None,
            "nce": rec["params"].get('COLLISION_ENERGY'),
            "instrument": rec["params"].get('INSTRUMENT_TYPE'),
            "smi_key": "C",  # dummy SMILES required by collate function
        }
        return {
            "mz": rec["mz"],
            "intensity": intensity,  # Now normalized to [0, 1]
            "title": rec["title"],
            "feature_id": rec["feature_id"],
            "meta": meta,
            "smiles": "C"  # dummy, not used
        }


def collate_with_feature_id(batch, spec_bins, formula_vocab, adduct_vocab, charge_vocab, fp_bits, seed):
    """Collate function that preserves feature_id."""
    out = collate_massspecgym(batch, spec_bins, formula_vocab, adduct_vocab, charge_vocab, fp_bits, seed=seed)
    out["feature_ids"] = [ex["feature_id"] for ex in batch]
    out["titles"] = [ex["title"] for ex in batch]
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser("Predict SMILES for each spectrum in MGF")
    ap.add_argument("--mgf", required=True, type=str, help="Input MGF file")
    ap.add_argument("--dreams-ckpt", required=True, type=str, help="DreaMS checkpoint path")
    ap.add_argument("--adapter-ckpt", required=True, type=str, help="Adapter checkpoint path")
    ap.add_argument("--candidates-json", required=True, type=str, help="JSON file with {feature_id: [candidate_smiles]}")
    ap.add_argument("--output", required=True, type=str, help="Output file (JSON or TSV)")
    ap.add_argument("--spec-bins", type=int, default=2048)
    ap.add_argument("--cond-dim", type=int, default=2048)
    ap.add_argument("--mapper-hidden", type=int, default=2048)
    ap.add_argument("--no-gaussian", action="store_true", help="Disable Gaussian uncertainty")
    ap.add_argument("--use-mapped", action="store_true", help="Use mapped embedding for query")
    ap.add_argument("--deterministic-map", action="store_true", help="Use deterministic mapping (mean)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--mol-space", type=str, default="chemberta", choices=["chemberta", "ecfp"])
    ap.add_argument("--chemberta-model", type=str, default="Derify/ChemBERTa_augmented_pubchem_13m")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of spectra to process")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--handle-duplicates", type=str, default="keep_first", 
                    choices=["keep_first", "combine"], 
                    help="How to handle duplicate feature_ids: keep_first (use first) or combine (merge candidates)")
    ap.add_argument("--top-k", type=int, default=1, help="Output top K predictions per spectrum")
    ap.add_argument("--no-normalize-intensities", action="store_true", 
                    help="Disable intensity normalization (use if MGF already has normalized intensities like MassSpecGym.mgf)")

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)

    print(f"[config] device={device}")
    print(f"[config] mol_space={args.mol_space}")
    print(f"[config] use_mapped={args.use_mapped}")
    print(f"[config] deterministic_map={args.deterministic_map}")

    # Load candidates JSON
    print(f"[candidates] Loading from {args.candidates_json}")
    with open(args.candidates_json, 'r') as f:
        candidates_dict = json.load(f)
    print(f"[candidates] Loaded {len(candidates_dict)} feature_id entries")

    # Build model
    model = build_model(args, device)
    embed_fn = build_mol_embed_fn(args, model, device)

    # Create dataset
    print(f"[dataset] Loading MGF from {args.mgf}")
    normalize_intensities = not args.no_normalize_intensities
    print(f"[dataset] Intensity normalization: {'enabled' if normalize_intensities else 'disabled'}")
    dataset = FeatureIDDataset(args.mgf, normalize_intensities=normalize_intensities)
    print(f"[dataset] Loaded {len(dataset)} spectra")

    # Handle duplicates if needed
    if args.handle_duplicates == "combine":
        # Group by feature_id and combine candidates
        feature_id_to_indices = {}
        for i, rec in enumerate(dataset._records):
            fid = rec["feature_id"]
            if fid:
                if fid not in feature_id_to_indices:
                    feature_id_to_indices[fid] = []
                feature_id_to_indices[fid].append(i)
        
        # Update candidates_dict to combine candidates for duplicates
        for fid, indices in feature_id_to_indices.items():
            if len(indices) > 1 and fid in candidates_dict:
                # Already have candidates, keep as is
                pass
        print(f"[dataset] Found {len([fid for fid, idxs in feature_id_to_indices.items() if len(idxs) > 1])} duplicate feature_ids")

    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_with_feature_id(
            b, args.spec_bins, 32, 16, 8, 2048, seed=args.seed
        ),
        num_workers=0,
        pin_memory=False,
    )

    # Precompute all candidate embeddings
    print("[embedding] Collecting unique candidate SMILES...")
    all_candidate_smiles = set()
    for fid, cand_list in candidates_dict.items():
        if isinstance(cand_list, list):
            all_candidate_smiles.update(cand_list)
    all_candidate_smiles = sorted(list(all_candidate_smiles))
    print(f"[embedding] Found {len(all_candidate_smiles)} unique candidate SMILES")

    print("[embedding] Computing candidate embeddings...")
    cand_embeddings = {}
    batch_size_emb = 512
    for i in tqdm(range(0, len(all_candidate_smiles), batch_size_emb), desc="Embedding candidates"):
        chunk = all_candidate_smiles[i:i+batch_size_emb]
        Z = embed_fn(chunk, device, bs=batch_size_emb)
        for smi, z in zip(chunk, Z):
            canon_smi = _canon_smi(smi) or smi
            cand_embeddings[canon_smi] = z
    print(f"[embedding] Computed embeddings for {len(cand_embeddings)} candidates")

    # Process spectra and predict
    print("[prediction] Processing spectra...")
    results = []
    total_processed = 0
    missing_candidates = 0
    missing_embeddings = 0

    for batch in tqdm(loader, desc="Predicting"):
        if args.limit and total_processed >= args.limit:
            break

        s = batch["spectra"].to(device)
        B = s.size(0)  # Get batch size first
        meta = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch["meta"].items()}
        feature_ids = batch["feature_ids"]
        titles = batch["titles"]

        # Get query embeddings
        # For inference, we need dummy SMILES in meta for chemberta forward pass
        # but we won't use z_m - we only need z_s and mu
        dummy_smiles = ["C"] * B
        meta_with_smiles = meta.copy()
        meta_with_smiles["smi_key"] = dummy_smiles
        
        z_s, z_m, z_hat, mu, lv = model(s, meta_with_smiles, None, inference=True)
        
        # Build query embeddings - must match candidate embedding space
        # Candidates are raw ChemBERTa embeddings (768D), so query should also be in that space
        # mu is already computed in the forward pass and is in ChemBERTa space (768D)
        if args.use_mapped:
            # Use mu (mapped spectrum embedding in ChemBERTa space)
            if args.deterministic_map or (lv is None):
                z_query = mu  # [B, 768] - matches raw ChemBERTa embeddings
            else:
                z_query = model.mapB.sample(mu, lv, deterministic=True)  # [B, 768]
        else:
            # When not using mapped, we still need to use mu to match candidate space
            # (z_s is 2048D but candidates are 768D, so we use mu which is 768D)
            if args.deterministic_map or (lv is None):
                z_query = mu  # [B, 768] - use mu to match candidate embeddings
            else:
                z_query = model.mapB.sample(mu, lv, deterministic=True)  # [B, 768]
        for i in range(B):
            if args.limit and total_processed >= args.limit:
                break

            feature_id = feature_ids[i]
            title = titles[i]
            
            # Handle None feature_id
            if feature_id is None:
                results.append({
                    "feature_id": None,
                    "title": title,
                    "predicted_smiles": None,
                    "top_k_smiles": [],
                    "top_k_scores": [],
                    "all_candidates": [],
                    "all_scores": [],
                    "num_candidates": 0,
                    "status": "no_feature_id"
                })
                total_processed += 1
                continue
            
            zq = z_query[i]  # [D]

            # Get candidates for this feature_id
            cand_list = candidates_dict.get(feature_id, [])
            # Ensure cand_list is a list
            if not isinstance(cand_list, list):
                cand_list = []
            if not cand_list:
                missing_candidates += 1
                results.append({
                    "feature_id": feature_id,
                    "title": title,
                    "predicted_smiles": None,
                    "top_k_smiles": [],
                    "top_k_scores": [],
                    "all_candidates": [],
                    "all_scores": [],
                    "num_candidates": 0,
                    "status": "no_candidates"
                })
                total_processed += 1
                continue

            # Get embeddings for candidates
            Z_candidates = []
            candidate_smiles = []
            for smi in cand_list:
                canon_smi = _canon_smi(smi) or smi
                z_cand = cand_embeddings.get(canon_smi)
                if z_cand is not None:
                    Z_candidates.append(z_cand)
                    candidate_smiles.append(canon_smi)
                else:
                    missing_embeddings += 1

            if not Z_candidates:
                results.append({
                    "feature_id": feature_id,
                    "title": title,
                    "predicted_smiles": None,
                    "top_k_smiles": [],
                    "top_k_scores": [],
                    "all_candidates": [],
                    "all_scores": [],
                    "num_candidates": len(cand_list),
                    "status": "no_embeddings"
                })
                total_processed += 1
                continue

            # Compute similarities
            Z_stack = torch.stack(Z_candidates, dim=0).to(device)  # [C, D]
            zq_norm = F.normalize(zq.unsqueeze(0), dim=-1)  # [1, D]
            Z_norm = F.normalize(Z_stack, dim=-1)  # [C, D]
            similarities = (zq_norm @ Z_norm.T).squeeze(0)  # [C]

            # Sort all candidates by similarity (descending)
            all_indices = torch.argsort(similarities, descending=True)
            all_smiles_sorted = [candidate_smiles[idx] for idx in all_indices]
            all_scores_sorted = [float(similarities[idx]) for idx in all_indices]

            # Get top-k for backward compatibility
            top_k = min(args.top_k, len(similarities))
            top_smiles = all_smiles_sorted[:top_k]
            top_scores = all_scores_sorted[:top_k]

            results.append({
                "feature_id": feature_id,
                "title": title,
                "predicted_smiles": all_smiles_sorted[0] if all_smiles_sorted else None,
                "top_k_smiles": top_smiles,
                "top_k_scores": top_scores,
                "all_candidates": all_smiles_sorted,  # All candidates sorted by score (descending)
                "all_scores": all_scores_sorted,  # All scores sorted (descending)
                "num_candidates": len(cand_list),
                "status": "success"
            })
            total_processed += 1

    print(f"\n[summary] Processed {total_processed} spectra")
    print(f"[summary] Missing candidates: {missing_candidates}")
    print(f"[summary] Missing embeddings: {missing_embeddings}")

    # Write output
    print(f"[output] Writing results to {args.output}")
    if args.output.endswith('.json'):
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
    elif args.output.endswith('.tsv'):
        import csv
        with open(args.output, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['feature_id', 'title', 'predicted_smiles', 'score', 'num_candidates', 'status'])
            for r in results:
                writer.writerow([
                    r['feature_id'],
                    r['title'],
                    r['predicted_smiles'] or '',
                    r['top_k_scores'][0] if r['top_k_scores'] else '',
                    r['num_candidates'],
                    r['status']
                ])
    else:
        # Default to JSON
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

    print(f"[done] Results written to {args.output}")


class SpecBridgePredictor:
    """
    Jupyter-friendly wrapper for SpecBridge SMILES prediction.
    
    Usage in Jupyter notebook:
    ```python
    from specbridge.eval.predict_smiles import SpecBridgePredictor
    
    predictor = SpecBridgePredictor(
        mgf="data/my_spectra.mgf",
        dreams_ckpt="DreaMS/dreams/models/pretrained/ssl_model.ckpt",
        adapter_ckpt="runs/msgym/SpecBridge_MSGYM_checkpoint.pt",
        candidates_json="data/candidates.json",
        use_mapped=True,
        deterministic_map=True,
        no_gaussian=True,
        batch_size=32,
        cond_dim=2048,
        mapper_hidden=2048,
        mol_space="chemberta",
        chemberta_model="Derify/ChemBERTa_augmented_pubchem_13m",
        top_k=5,
        seed=1234,
        cpu=False
    )
    
    # Run prediction
    results = predictor.predict()
    
    # Save results
    predictor.save_results(results, "predictions.json")
    ```
    """
    
    def __init__(
        self,
        mgf: str,
        dreams_ckpt: str,
        adapter_ckpt: str,
        candidates_json: str,
        spec_bins: int = 2048,
        cond_dim: int = 2048,
        mapper_hidden: int = 2048,
        n_blocks=8,  # Number of blocks in the mapper
        no_gaussian: bool = True,
        use_mapped: bool = True,
        deterministic_map: bool = True,
        batch_size: int = 32,
        mol_space: str = "chemberta",
        chemberta_model: str = "Derify/ChemBERTa_augmented_pubchem_13m",
        limit: Optional[int] = None,
        seed: int = 1234,
        cpu: bool = False,
        handle_duplicates: str = "keep_first",
        top_k: int = 1,
        no_normalize_intensities: bool = False,
    ):
        """
        Initialize the SpecBridge predictor.
        
        Args:
            mgf: Path to input MGF file with mass spectra
            dreams_ckpt: Path to DreaMS backbone checkpoint (ssl_model.ckpt)
            adapter_ckpt: Path to trained adapter checkpoint (.pt file)
            candidates_json: Path to JSON file with {feature_id: [candidate_smiles]}
            spec_bins: Number of bins for spectrum binning (default: 2048)
            cond_dim: Conditioning dimension (default: 2048)
            mapper_hidden: Mapper hidden dimension (default: 2048)
            no_gaussian: Disable Gaussian uncertainty (default: True)
            use_mapped: Use mapped embedding for query (default: True)
            deterministic_map: Use deterministic mapping/mean (default: True)
            batch_size: Batch size for processing (default: 32)
            mol_space: Molecule embedding space - "chemberta" or "ecfp" (default: "chemberta")
            chemberta_model: HuggingFace ChemBERTa model name (default: "Derify/ChemBERTa_augmented_pubchem_13m")
            limit: Limit number of spectra to process (default: None = all)
            seed: Random seed (default: 1234)
            cpu: Force CPU usage even if GPU available (default: False)
            handle_duplicates: How to handle duplicate feature_ids - "keep_first" or "combine" (default: "keep_first")
            top_k: Number of top predictions to return per spectrum (default: 1)
            no_normalize_intensities: Disable intensity normalization (default: False)
        """
        # Create args object matching main() expectations
        self.args = argparse.Namespace(
            mgf=mgf,
            dreams_ckpt=dreams_ckpt,
            adapter_ckpt=adapter_ckpt,
            candidates_json=candidates_json,
            spec_bins=spec_bins,
            cond_dim=cond_dim,
            mapper_hidden=mapper_hidden,
            n_blocks=n_blocks,
            no_gaussian=no_gaussian,
            use_mapped=use_mapped,
            deterministic_map=deterministic_map,
            batch_size=batch_size,
            mol_space=mol_space,
            chemberta_model=chemberta_model,
            limit=limit,
            seed=seed,
            cpu=cpu,
            handle_duplicates=handle_duplicates,
            top_k=top_k,
            no_normalize_intensities=no_normalize_intensities,
        )
        
        # Setup device and seed
        self.device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
        set_seed(seed)
        
        print(f"[config] device={self.device}")
        print(f"[config] mol_space={mol_space}")
        print(f"[config] use_mapped={use_mapped}")
        print(f"[config] deterministic_map={deterministic_map}")
        print(f"[config] n_blocks={n_blocks}")
        
        # Will be initialized in predict()
        self.model = None
        self.embed_fn = None
        self.candidates_dict = None
        
    def _load_candidates(self):
        """Load candidates JSON file."""
        print(f"[candidates] Loading from {self.args.candidates_json}")
        with open(self.args.candidates_json, 'r') as f:
            self.candidates_dict = json.load(f)
        print(f"[candidates] Loaded {len(self.candidates_dict)} feature_id entries")
    
    def _build_model(self):
        """Build and load the SpecBridge model."""
        print("[model] Building SpecBridge model...")
        self.model = build_model(self.args, self.device)
        self.embed_fn = build_mol_embed_fn(self.args, self.model, self.device)
        print("[model] Model ready")
    
    @torch.no_grad()
    def predict(self) -> List[Dict]:
        """
        Run prediction on all spectra in the MGF file.
        
        Returns:
            List of prediction results, where each result is a dict with:
                - feature_id: Feature ID from MGF
                - title: Spectrum title
                - predicted_smiles: Top predicted SMILES
                - top_k_smiles: List of top-K predicted SMILES
                - top_k_scores: List of top-K similarity scores
                - all_candidates: All candidates sorted by score (descending)
                - all_scores: All scores sorted (descending)
                - num_candidates: Number of candidates available
                - status: "success", "no_candidates", "no_embeddings", etc.
        """
        # Load candidates if not already loaded
        if self.candidates_dict is None:
            self._load_candidates()
        
        # Build model if not already built
        if self.model is None:
            self._build_model()
        
        # Create dataset
        print(f"[dataset] Loading MGF from {self.args.mgf}")
        normalize_intensities = not self.args.no_normalize_intensities
        print(f"[dataset] Intensity normalization: {'enabled' if normalize_intensities else 'disabled'}")
        dataset = FeatureIDDataset(self.args.mgf, normalize_intensities=normalize_intensities)
        print(f"[dataset] Loaded {len(dataset)} spectra")
        
        # Handle duplicates if needed
        if self.args.handle_duplicates == "combine":
            feature_id_to_indices = {}
            for i, rec in enumerate(dataset._records):
                fid = rec["feature_id"]
                if fid:
                    if fid not in feature_id_to_indices:
                        feature_id_to_indices[fid] = []
                    feature_id_to_indices[fid].append(i)
            
            for fid, indices in feature_id_to_indices.items():
                if len(indices) > 1 and fid in self.candidates_dict:
                    pass  # Already have candidates, keep as is
            print(f"[dataset] Found {len([fid for fid, idxs in feature_id_to_indices.items() if len(idxs) > 1])} duplicate feature_ids")
        
        # Create dataloader
        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_with_feature_id(
                b, self.args.spec_bins, 32, 16, 8, 2048, seed=self.args.seed
            ),
            num_workers=0,
            pin_memory=False,
        )
        
        # Precompute all candidate embeddings
        print("[embedding] Collecting unique candidate SMILES...")
        all_candidate_smiles = set()
        for fid, cand_list in self.candidates_dict.items():
            if isinstance(cand_list, list):
                for item in cand_list:
                    if isinstance(item, list):
                        all_candidate_smiles.update(item)
                    else:
                        all_candidate_smiles.add(item)
            else:
                all_candidate_smiles.add(cand_list)
        all_candidate_smiles = sorted(list(all_candidate_smiles))
        print(f"[embedding] Found {len(all_candidate_smiles)} unique candidate SMILES")
        
        print("[embedding] Computing candidate embeddings...")
        cand_embeddings = {}
        batch_size_emb = 512
        for i in tqdm(range(0, len(all_candidate_smiles), batch_size_emb), desc="Embedding candidates"):
            chunk = all_candidate_smiles[i:i+batch_size_emb]
            Z = self.embed_fn(chunk, self.device, bs=batch_size_emb)
            for smi, z in zip(chunk, Z):
                canon_smi = _canon_smi(smi) or smi
                cand_embeddings[canon_smi] = z
        print(f"[embedding] Computed embeddings for {len(cand_embeddings)} candidates")
        
        # Process spectra and predict
        print("[prediction] Processing spectra...")
        results = []
        total_processed = 0
        missing_candidates = 0
        missing_embeddings = 0
        
        for batch in tqdm(loader, desc="Predicting"):
            if self.args.limit and total_processed >= self.args.limit:
                break
            
            s = batch["spectra"].to(self.device)
            B = s.size(0)
            meta = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch["meta"].items()}
            feature_ids = batch["feature_ids"]
            titles = batch["titles"]
            
            # Get query embeddings
            dummy_smiles = ["C"] * B
            meta_with_smiles = meta.copy()
            meta_with_smiles["smi_key"] = dummy_smiles

            print(meta_with_smiles) # TESTING,  DELETE LATER
            
            z_s, z_m, z_hat, mu, lv = self.model(s, meta_with_smiles, None, inference=True)
            
            # Build query embeddings
            if self.args.use_mapped:
                if self.args.deterministic_map or (lv is None):
                    z_query = mu
                else:
                    z_query = self.model.mapB.sample(mu, lv, deterministic=True)
            else:
                if self.args.deterministic_map or (lv is None):
                    z_query = mu
                else:
                    z_query = self.model.mapB.sample(mu, lv, deterministic=True)
            
            for i in range(B):
                if self.args.limit and total_processed >= self.args.limit:
                    break
                
                feature_id = feature_ids[i]
                title = titles[i]
                
                # Handle None feature_id
                if feature_id is None:
                    results.append({
                        "feature_id": None,
                        "title": title,
                        "predicted_smiles": None,
                        "top_k_smiles": [],
                        "top_k_scores": [],
                        "all_candidates": [],
                        "all_scores": [],
                        "num_candidates": 0,
                        "status": "no_feature_id"
                    })
                    total_processed += 1
                    continue
                
                zq = z_query[i]
                
                # Get candidates for this feature_id
                cand_list = self.candidates_dict.get(feature_id, [])
                if not isinstance(cand_list, list):
                    cand_list = []
                if not cand_list:
                    missing_candidates += 1
                    results.append({
                        "feature_id": feature_id,
                        "title": title,
                        "predicted_smiles": None,
                        "top_k_smiles": [],
                        "top_k_scores": [],
                        "all_candidates": [],
                        "all_scores": [],
                        "num_candidates": 0,
                        "status": "no_candidates"
                    })
                    total_processed += 1
                    continue
                
                # Get embeddings for candidates
                Z_candidates = []
                candidate_smiles = []
                for smi in cand_list:
                    canon_smi = _canon_smi(smi) or smi
                    z_cand = cand_embeddings.get(canon_smi)
                    if z_cand is not None:
                        Z_candidates.append(z_cand)
                        candidate_smiles.append(canon_smi)
                    else:
                        missing_embeddings += 1
                
                if not Z_candidates:
                    results.append({
                        "feature_id": feature_id,
                        "title": title,
                        "predicted_smiles": None,
                        "top_k_smiles": [],
                        "top_k_scores": [],
                        "all_candidates": [],
                        "all_scores": [],
                        "num_candidates": len(cand_list),
                        "status": "no_embeddings"
                    })
                    total_processed += 1
                    continue
                
                # Compute similarities
                Z_stack = torch.stack(Z_candidates, dim=0).to(self.device)
                zq_norm = F.normalize(zq.unsqueeze(0), dim=-1)
                Z_norm = F.normalize(Z_stack, dim=-1)
                similarities = (zq_norm @ Z_norm.T).squeeze(0)
                
                # Sort all candidates by similarity (descending)
                all_indices = torch.argsort(similarities, descending=True)
                all_smiles_sorted = [candidate_smiles[idx] for idx in all_indices]
                all_scores_sorted = [float(similarities[idx]) for idx in all_indices]
                
                # Get top-k
                top_k = min(self.args.top_k, len(similarities))
                top_smiles = all_smiles_sorted[:top_k]
                top_scores = all_scores_sorted[:top_k]
                
                results.append({
                    "feature_id": feature_id,
                    "title": title,
                    "predicted_smiles": all_smiles_sorted[0] if all_smiles_sorted else None,
                    "top_k_smiles": top_smiles,
                    "top_k_scores": top_scores,
                    "all_candidates": all_smiles_sorted,
                    "all_scores": all_scores_sorted,
                    "num_candidates": len(cand_list),
                    "status": "success"
                })
                total_processed += 1
        
        print(f"\n[summary] Processed {total_processed} spectra")
        print(f"[summary] Missing candidates: {missing_candidates}")
        print(f"[summary] Missing embeddings: {missing_embeddings}")
        
        return results
    
    def save_results(self, results: List[Dict], output_path: str):
        """
        Save prediction results to file.
        
        Args:
            results: List of prediction results from predict()
            output_path: Output file path (.json or .tsv)
        """
        print(f"[output] Writing results to {output_path}")
        if output_path.endswith('.json'):
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        elif output_path.endswith('.tsv'):
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['feature_id', 'title', 'predicted_smiles', 'score', 'num_candidates', 'status'])
                for r in results:
                    writer.writerow([
                        r['feature_id'],
                        r['title'],
                        r['predicted_smiles'] or '',
                        r['top_k_scores'][0] if r['top_k_scores'] else '',
                        r['num_candidates'],
                        r['status']
                    ])
        else:
            # Default to JSON
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        print(f"[done] Results written to {output_path}")

if __name__ == "__main__":
    main()