"""
Notebook-friendly SpecBridge predictor.
--------------------------------------
Minimal helper to load trained checkpoints and score candidate SMILES directly
from Python (e.g., inside a Jupyter notebook) without invoking the CLI script.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, Sequence, Set

import torch
import torch.nn.functional as F

from specbridge.eval.predict_smiles import (
    _canon_smi,
    build_model,
    build_mol_embed_fn,
    collate_with_feature_id,
)
from specbridge.utils.common import set_seed

DUMMY_SMILES = "C"


class SpecBridgeNotebookPredictor:
    """
    Lightweight wrapper for notebook inference.

    Usage:
        predictor = SpecBridgeNotebookPredictor(
            dreams_ckpt="runs/msgym/ssl_model.ckpt",
            adapter_ckpt="runs/msgym/checkpoint.ckpt",
            device="cuda"  # or "cpu"
        )
        predictor.embed_candidates(["CCO", "CCN"])
        # Provide your own spectrum arrays (1D m/z and intensity values)
        result = predictor.predict(your_mz_values, your_intensity_values, top_k=3)
    
    Args:
        dreams_ckpt: Path to the DreaMS checkpoint (ssl_model.ckpt).
        adapter_ckpt: Path to the SpecBridge adapter checkpoint.
        device: Target device; defaults to CUDA if available.
        spec_bins: Number of spectral bins (matches training setup).
        cond_dim: SpecBridge conditioning dimension.
        mapper_hidden: Hidden size for the mapper network.
        n_blocks: Residual mapper blocks (matches training default of 8).
        mol_space: Molecule embedding space (ChemBERTa supported).
        chemberta_model: Hugging Face ChemBERTa model name.
        no_gaussian: Disable Gaussian sampling in the mapper.
        formula_vocab: One-hot vocab size for formulas; keep defaults for pretrained checkpoints.
        adduct_vocab: One-hot vocab size for adducts; keep defaults for pretrained checkpoints.
        charge_vocab: One-hot vocab size for charges; keep defaults for pretrained checkpoints.
        fp_bits: Fingerprint length used for placeholder molecule features.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        dreams_ckpt: str,
        adapter_ckpt: str,
        *,
        device: str | None = None,
        spec_bins: int = 2048,
        cond_dim: int = 2048,
        mapper_hidden: int = 2048,
        n_blocks: int = 8,
        mol_space: str = "chemberta",
        chemberta_model: str = "Derify/ChemBERTa_augmented_pubchem_13m",
        no_gaussian: bool = False,
        seed: int = 1234,
        formula_vocab: int = 32,
        adduct_vocab: int = 16,
        charge_vocab: int = 8,
        fp_bits: int = 2048,
    ):
        set_seed(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.formula_vocab = formula_vocab
        self.adduct_vocab = adduct_vocab
        self.charge_vocab = charge_vocab
        self.fp_bits = fp_bits
        self.seed = seed
        self.spec_bins = spec_bins

        args = SimpleNamespace(
            dreams_ckpt=dreams_ckpt,
            adapter_ckpt=adapter_ckpt,
            spec_bins=spec_bins,
            cond_dim=cond_dim,
            mapper_hidden=mapper_hidden,
            mol_space=mol_space,
            chemberta_model=chemberta_model,
            no_gaussian=no_gaussian,
            n_blocks=n_blocks,
        )
        self.model = build_model(args, self.device)
        self.embed_fn = build_mol_embed_fn(args, self.model, self.device)

        self._cand_smiles: List[str] | None = None
        self._cand_embeddings: torch.Tensor | None = None

    def embed_candidates(self, candidates: Sequence[str], *, batch_size: int = 512) -> List[str]:
        """Pre-compute and cache candidate embeddings (canonicalized)."""
        uniq: List[str] = []
        seen: Set[str] = set()
        for smi in candidates:
            canon = _canon_smi(smi) or smi
            if canon not in seen:
                uniq.append(canon)
                seen.add(canon)
        # Reuse cached embeddings if candidate list is unchanged.
        if self._cand_smiles == uniq:
            return uniq
        if not uniq:
            self._cand_smiles = []
            self._cand_embeddings = None
            return []

        Z = F.normalize(self.embed_fn(uniq, self.device, bs=batch_size), dim=-1)
        self._cand_smiles = uniq
        self._cand_embeddings = Z
        return uniq

    def _compute_mapped(self, mu: torch.Tensor, lv: torch.Tensor | None, deterministic_map: bool) -> torch.Tensor:
        if lv is None or deterministic_map:
            return mu
        return self.model.mapB.sample(mu, lv, deterministic=False)

    def _make_batch(
        self,
        mz: Sequence[float] | torch.Tensor,
        intensity: Sequence[float] | torch.Tensor,
        *,
        normalize_intensities: bool,
        title: str,
    ) -> Dict[str, torch.Tensor]:
        mz_t = torch.as_tensor(mz, dtype=torch.float32)
        inten_t = torch.as_tensor(intensity, dtype=torch.float32)
        if normalize_intensities and inten_t.numel() > 0:
            max_int = inten_t.max()
            if max_int > 0:
                inten_t = inten_t / max_int

        rec = {
            "mz": mz_t,
            "intensity": inten_t,
            "title": title,
            "feature_id": title,
            "meta": {
                "formula_idx": None,
                "adduct_idx": None,
                "charge_idx": None,
                "nce": None,
                "instrument": None,
                "smi_key": DUMMY_SMILES,
            },
            "smiles": DUMMY_SMILES,
        }
        return collate_with_feature_id(
            [rec],
            self.spec_bins,
            self.formula_vocab,
            self.adduct_vocab,
            self.charge_vocab,
            self.fp_bits,
            seed=self.seed,
        )

    @torch.no_grad()
    def predict(
        self,
        mz: Sequence[float] | torch.Tensor,
        intensity: Sequence[float] | torch.Tensor,
        *,
        candidates: Sequence[str] | None = None,
        top_k: int = 5,
        normalize_intensities: bool = True,
        title: str = "query_0",
        use_mapped: bool = True,
        deterministic_map: bool = True,
    ) -> Dict[str, object]:
        """
        Score candidate SMILES for a single spectrum.

        Args:
            mz: iterable of m/z values.
            intensity: iterable of intensities matching ``mz``.
            candidates: list of candidate SMILES (optional if already cached via ``embed_candidates``).
            top_k: number of top predictions to return.
            normalize_intensities: divide intensities by max value before binning.
            title: identifier for the spectrum (used in outputs only).
            use_mapped: when False, try spectrum-space embedding if its dimension matches
                candidate embeddings; otherwise fall back to the mapped embedding.
            deterministic_map: controls sampling in the mapper; also affects the fallback
                path when ``use_mapped`` is False but a mapped embedding is required.
        """
        if candidates is not None:
            self.embed_candidates(candidates)
        if self._cand_smiles is None:
            raise ValueError("No candidates provided. Call embed_candidates() or pass candidates to predict().")
        if len(self._cand_smiles) == 0:
            return {
                "title": title,
                "predicted_smiles": None,
                "top_k_smiles": [],
                "top_k_scores": [],
                "all_candidates": [],
                "all_scores": [],
                "status": "no_candidates",
            }
        if self._cand_embeddings is None:
            raise ValueError("Candidate embeddings are missing. Call embed_candidates() first.")

        batch = self._make_batch(mz, intensity, normalize_intensities=normalize_intensities, title=title)
        spectra = batch["spectra"].to(self.device)
        meta = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch["meta"].items()}
        meta["smi_key"] = meta.get("smi_key", [DUMMY_SMILES])

        cand_emb = self._cand_embeddings.to(self.device)
        z_s, z_m, z_hat, mu, lv = self.model(spectra, meta, None, inference=True)
        if use_mapped:
            z_query = self._compute_mapped(mu, lv, deterministic_map)
        else:
            # Only fall back to z_s (spectrum space) if it already matches
            # the candidate embedding dimensionality (i.e., is compatible with the candidate space).
            if z_s.shape[-1] == cand_emb.shape[-1]:
                z_query = z_s
            else:
                z_query = self._compute_mapped(mu, lv, deterministic_map)

        sims = (F.normalize(z_query, dim=-1) @ cand_emb.T).squeeze(0)
        order = torch.argsort(sims, descending=True)
        k = min(top_k, len(order))
        top_smiles = [self._cand_smiles[i] for i in order[:k]]
        top_scores = [float(sims[i]) for i in order[:k]]
        all_scores = [float(sims[i]) for i in order]

        return {
            "title": title,
            "predicted_smiles": top_smiles[0] if top_smiles else None,
            "top_k_smiles": top_smiles,
            "top_k_scores": top_scores,
            "all_candidates": [self._cand_smiles[i] for i in order],
            "all_scores": all_scores,
            "status": "success",
        }
