from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# 20 canonical AAs + ambiguity code in ESM2's vocab
ALLOWED = set("ACDEFGHIKLMNPQRSTVWY") | set("XBZU")


def sanitize_sequence(seq: str) -> str:
    """Uppercase, strip whitespace, map unknown chars to 'X' with a log"""
    s = seq.strip().upper()
    bad = sorted({c for c in s if c not in ALLOWED})
    if bad:
        logger.warning("mapping non-strandard residues %s to 'X'", bad)
        for c in bad:
            s = s.replace(c, "X")
    return s


class ESM2Source:
    """A HuggingFace ESM-2 checkpoint satisfying the EmbeddingSource protocol.

    The model is lazy: __init__ stores config only; weights load on first
    embed() call. dim and cache_id work without ever touching weights.
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        layer: int | None = None,  # index into hidden states, None for last layer
        device: str | None = None,  # None = auto (cuda if available, else cpu)
    ) -> None:
        self.model_name = model_name
        self.layer = layer
        self.device = device
        self._dim: int | None = None
        self._tok = None
        self._model = None
        self._cfg = None

    # EmbeddingSource: id
    @property
    def cache_id(self) -> str:
        # fully deterministic from constructor args
        return f"hf|{self.model_name}|layer={self.layer if self.layer is not None else 'last'}"

    @property
    def hf_config(self):
        if self._cfg is None:
            from transformers import AutoConfig

            self._cfg = AutoConfig.from_pretrained(self.model_name)
        return self._cfg

    @property
    def dim(self) -> int:
        return int(self.hf_config.hidden_size)

    # lazy loading

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        config = AutoConfig.from_pretrained(self.model_name)
        n = config.num_hidden_layers
        if self.layer is not None and not 0 <= self.layer <= n:
            raise ValueError(
                f"layer={self.layer} invalid for {self.model_name}: "
                f"hidden_states has indices 0..{n} (0 = input embeddings, {n} = last layer)"
            )
        resolved = (
            torch.device(self.device)
            if self.device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        dtype = torch.bfloat16 if resolved.type == "cuda" else torch.float32
        logger.info("loading %s on %s (%s)", self.model_name, resolved, dtype)
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = (
            AutoModel.from_pretrained(self.model_name).to(device=resolved, dtype=dtype).eval()
        )
        self._resolved_device = resolved
        self._torch = torch

    # embedding source

    def embed(self, seqs: list[str]) -> dict[str, np.ndarray]:
        """Sequences -> {original_seq: (L, D) float16}"""
        if not seqs:
            return {}
        self._ensure_loaded()
        torch = self._torch

        clean = [sanitize_sequence(s) for s in seqs]
        max_pos = getattr(self._model.config, "max_position_embeddings", None)
        if max_pos is not None:
            for s in clean:
                if len(s) + 2 > max_pos:  # +2 for BOS/EOS tokens
                    raise ValueError(
                        f"sequence of length {len(s)} exceeds {self.model_name} "
                        f"capacity ({max_pos - 2} residues). Truncating would violate the "
                        f"alignment clause (L == len(seq), always) — chunk externally or "
                        f"pick a longer-context model."
                    )
        self._ensure_loaded()
        enc = self._tok(clean, return_tensors="pt", padding=True, add_special_tokens=True)
        input_ids = enc["input_ids"].to(self._resolved_device)
        attention = enc["attention_mask"].to(self._resolved_device)
        with torch.inference_mode():
            out = self._model(
                input_ids=input_ids,
                attention_mask=attention,
                output_hidden_states=True,
            )
        hidden = out.hidden_states[self.layer if self.layer is not None else -1]

        # strip BOS and EOS
        result: dict[str, np.ndarray] = {}
        for i, original in enumerate(seqs):
            vec = (
                hidden[i, 1 : 1 + len(clean[i]), :]
                .to(torch.float32)
                .cpu()
                .numpy()
                .astype(np.float16)
            )
            result[original] = vec  # keyed to non sanitized caller
        return result


class SequenceTooLong(ValueError):
    """Raised instead of truncating: silent truncation would violate the
    alignment clause (L == len(seq)) that the whole harness trusts."""
