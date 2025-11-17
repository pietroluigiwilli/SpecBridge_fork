# specbridge/models/generatorT5.py
from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # optional: convert SELFIES <-> SMILES at inference if you choose that target space
    import selfies as sf
    _HAS_SELFIES = True
except Exception:
    _HAS_SELFIES = False

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
__all__ = [
    "SoftPromptT5",
    "build_generator",
    "cond_from_spectra",
    "cond_from_smiles_via_chemberta",
]


def _unit_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


class SoftPromptT5(nn.Module):
    """
    A tiny conditioning head that maps a continuous condition vector z (e.g., ChemBERTa
    embedding or your mapped spectrum embedding mu) into K 'virtual' encoder tokens
    (soft prompt) in T5 hidden space, then concatenates them to the encoder inputs.

    Only the mapper is trained by default; the LM stays frozen unless freeze_lm=False.
    """
    def __init__(
        self,
        lm_name: str = "laituan245/molt5-small",
        cond_dim: int = 2048,
        prompt_len: int = 10,
        hidden: int = 0,
        dropout: float = 0.0,
        freeze_lm: bool = True,
        use_selfies: bool = False,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(lm_name, use_fast=True, **(tokenizer_kwargs or {}))
        
        # T5 prefers right padding
        try:
            self.tokenizer.padding_side = "right"
        except Exception:
            pass
        if self.tokenizer.pad_token is None:
            # T5 usually uses </s> as both eos and pad in practice
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token


        self.lm = AutoModelForSeq2SeqLM.from_pretrained(lm_name)
        self.lm.eval()  # inference-safe default

        if freeze_lm:
            for p in self.lm.parameters():
                p.requires_grad = False

        self.use_selfies = use_selfies
        self.d_model = int(self.lm.config.d_model)
        self.prompt_len = int(prompt_len)
        self.cond_dim = int(cond_dim)

        # z -> (K * d_model) -> reshape to [B, K, d_model]
        if hidden and hidden > 0:
            self.mapper = nn.Sequential(
                nn.Linear(cond_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, prompt_len * self.d_model),
            )
        else:
            self.mapper = nn.Linear(cond_dim, prompt_len * self.d_model)

        # A small layernorm helps stabilize magnitude of soft prompts
        self.prompt_ln = nn.LayerNorm(self.d_model)

    # -----------------------
    # Encoding utilities
    # -----------------------
    def _cond_to_soft_prompt(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, cond_dim] (should be on same device as model)
        returns: soft_prompt [B, K, d_model]
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        z = _unit_normalize(z)
        h = self.mapper(z)                                   # [B, K*d]
        h = h.view(z.size(0), self.prompt_len, self.d_model) # [B, K, d]
        h = self.prompt_ln(h)
        return h

    def _build_encoder_embeds(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        soft_prompt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepend soft_prompt to token embeddings; extend attention mask with ones.
        """
        # Embed tokens with T5 shared embeddings
        # Encoder uses lm.get_encoder(); both share lm.shared embedding table.
        encoder = self.lm.get_encoder()
        embed_tokens = getattr(encoder, "embed_tokens", None)
        if embed_tokens is None:
            embed_tokens = self.lm.shared  # fallback
        tok_emb = embed_tokens(input_ids)  # [B, T, d_model]

        # Concat along sequence dimension
        enc_embeds = torch.cat([soft_prompt, tok_emb], dim=1)  # [B, K+T, d]
        B, K, _ = soft_prompt.shape
        new_mask = torch.cat(
            [torch.ones((B, K), device=attention_mask.device, dtype=attention_mask.dtype), attention_mask],
            dim=1,
        )  # [B, K+T]
        return enc_embeds, new_mask

    # -----------------------
    # Forward (teacher forcing)
    # -----------------------
    def forward(
        self,
        cond_vecs: torch.Tensor,              # [B, cond_dim]
        tgt_text: List[str],                  # SMILES or SELFIES
        src_text: Optional[List[str]] = None, # optional textual prefix ("generate molecule:")
        max_src_len: int = 64,
        max_tgt_len: int = 128,
        label_smoothing: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Teacher-forced NLL loss for fine-tuning the soft prompt on paired data.
        """
        device = next(self.parameters()).device
        B = cond_vecs.size(0)

        # Prepare target tokens
        if self.use_selfies:
            assert _HAS_SELFIES, "Install `selfies` to use use_selfies=True."
            tgt_text = [sf.encoder(s) if (s and not sf.is_valid_selfies(s)) else s for s in tgt_text]

        tgt = self.tokenizer(
            tgt_text,
            padding=True,
            truncation=True,
            max_length=max_tgt_len,
            return_tensors="pt",
        ).to(device)

        # Optional source text prompt (can be empty)
        src_text = src_text or ["generate molecule"] * B
        src = self.tokenizer(
            src_text,
            padding=True,
            truncation=True,
            max_length=max_src_len,
            return_tensors="pt",
        ).to(device)

        soft_prompt = self._cond_to_soft_prompt(cond_vecs.to(device))  # [B, K, d]
        enc_embeds, enc_mask = self._build_encoder_embeds(
            input_ids=src["input_ids"], attention_mask=src["attention_mask"], soft_prompt=soft_prompt
        )

        # labels: shift handled internally by HF
        outputs = self.lm(
            inputs_embeds=enc_embeds,
            attention_mask=enc_mask,
            labels=tgt["input_ids"],
        )
        loss = outputs.loss
        if label_smoothing and label_smoothing > 0:
            # Optional smoothing (simple cross-entropy smoothing on logits)
            logits = outputs.logits  # [B, T, V]
            ls = _label_smooth_loss(logits, tgt["input_ids"], smoothing=label_smoothing)
            loss = (loss + ls) * 0.5

        return {"loss": loss, "nll": outputs.loss.detach()}

    # -----------------------
    # Generation
    # -----------------------
    @torch.no_grad()
    def generate(
        self,
        cond_vecs: torch.Tensor,              # [B, cond_dim]
        src_text: Optional[List[str]] = None, # optional textual prefix
        max_src_len: int = 32,
        max_new_tokens: int = 128,
        num_beams: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        num_return_sequences: int = 1,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
    ) -> List[str]:
        """
        Generate SMILES/SELFIES strings conditioned on cond_vecs via soft prompt.
        """
        device = next(self.parameters()).device
        B = cond_vecs.size(0)

        src_text = src_text or ["generate molecule"] * B
        src = self.tokenizer(
            src_text,
            padding=True,
            truncation=True,
            max_length=max_src_len,
            return_tensors="pt",
        ).to(device)

        soft_prompt = self._cond_to_soft_prompt(cond_vecs.to(device))  # [B, K, d]
        enc_embeds, enc_mask = self._build_encoder_embeds(
            input_ids=src["input_ids"], attention_mask=src["attention_mask"], soft_prompt=soft_prompt
        )

        # Many HF seq2seq models accept inputs_embeds directly in generate.
        # If not, we build encoder_outputs manually as a fallback.
        try:
            gen_ids = self.lm.generate(
                inputs_embeds=enc_embeds,
                attention_mask=enc_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
            )
        except TypeError:
            # Fallback: run encoder to get encoder_outputs
            encoder = self.lm.get_encoder()
            enc_out = encoder(inputs_embeds=enc_embeds, attention_mask=enc_mask, return_dict=True)
            gen_ids = self.lm.generate(
                encoder_outputs=enc_out,
                attention_mask=enc_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
            )

        texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        if self.use_selfies and _HAS_SELFIES:
            # Convert SELFIES → SMILES
            out = []
            for t in texts:
                try:
                    out.append(sf.decoder(t))
                except Exception:
                    out.append(t)  # fallback: raw text
            return out
        return texts


def _label_smooth_loss(logits: torch.Tensor, target_ids: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
    """
    Simple label-smoothing CE for seq2seq. Ignores padding (tokenizer.pad_token_id).
    """
    V = logits.size(-1)
    logprobs = F.log_softmax(logits, dim=-1)
    with torch.no_grad():
        true_dist = torch.zeros_like(logprobs)
        true_dist.fill_(smoothing / (V - 1))
        # Mask pad positions
        pad_id = -100  # HF uses -100 in labels after internal shift (when passed as labels)
    # We need labels aligned with logits; callers pass target_ids before internal shift, so
    # only use this smoothing when we call with outputs.logits (already aligned inside forward).
    # Here, assume caller passes the right shapes (used only inside forward()).
    true_dist.scatter_(-1, target_ids.unsqueeze(-1).clamp_min(0), 1.0 - smoothing)
    loss = (-true_dist * logprobs).sum(dim=-1)
    # ignore label positions that were padding in HF (-100); best-effort mask:
    mask = (target_ids != -100).float()
    loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
    return loss


# ------------------------------------------------------------------------------------
# Convenience helpers to get condition vectors from *your* SpecBridge model components
# ------------------------------------------------------------------------------------

@torch.no_grad()
def cond_from_smiles_via_chemberta(smiles: List[str], model_with_chemberta, device: torch.device) -> torch.Tensor:
    """
    Uses your existing training model (e.g., DreamsToMolCondition) method `_chemberta_embed`
    to produce normalized condition vectors from SMILES strings.

    Returns: [N, D] tensor on `device`.
    """
    z = model_with_chemberta._chemberta_embed(smiles, device)  # expected [N, hidden] or [N, cond_dim] if projected
    # If your training model has a chem_proj (as in your code), apply it to match cond_dim.
    if hasattr(model_with_chemberta, "chem_proj") and isinstance(model_with_chemberta.chem_proj, nn.Linear):
        z = model_with_chemberta.chem_proj(z.to(device))
    z = _unit_normalize(z.to(device))
    return z


@torch.no_grad()
def cond_from_spectra(
    spectra_binned: torch.Tensor,
    meta: Dict[str, Any],
    dreams_to_mol_model,           # your trained DreamsToMolCondition
    deterministic: bool = True,
) -> torch.Tensor:
    """
    Maps spectra to the *mapped* molecule embedding space using your trained model.
    Returns normalized mu (or a sample if Gaussian and deterministic=False).
    """
    device = next(dreams_to_mol_model.parameters()).device
    z_s, z_m, z_hat, mu_s, lv_s = dreams_to_mol_model(
        spectra_binned.to(device),
        {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in meta.items()},
        None,
        inference=True
    )
    if hasattr(dreams_to_mol_model, "mapB") and (not deterministic) and (lv_s is not None):
        z = dreams_to_mol_model.mapB.sample(mu_s, lv_s, deterministic=False)
    else:
        z = mu_s
    return _unit_normalize(z)


def build_generator(
    lm_name: str,
    cond_dim: int,
    prompt_len: int = 10,
    hidden: int = 0,
    freeze_lm: bool = True,
    use_selfies: bool = False,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
) -> SoftPromptT5:
    """
    Factory to create the generator with your chosen LM and conditioning size.
    """
    gen = SoftPromptT5(
        lm_name=lm_name,
        cond_dim=cond_dim,
        prompt_len=prompt_len,
        hidden=hidden,
        freeze_lm=freeze_lm,
        use_selfies=use_selfies,
        tokenizer_kwargs=tokenizer_kwargs,
    )
    return gen
