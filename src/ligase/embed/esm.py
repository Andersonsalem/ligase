from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ligase.embed.batching import MAX_TOKENS_PER_BATCH
from ligase.embed.batching import plan_batches as _plan_batches

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


@dataclass(frozen=True)
class _Loaded:
    """Everything embed () needs once the weights are on the device"""

    model: Any
    tok: Any
    device: torch.device


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
        self._cfg: Any | None = None
        self._loaded: _Loaded | None = None

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
    def _load(self) -> _Loaded:
        if self._loaded is not None:
            return self._loaded
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
        tok = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name).to(device=resolved, dtype=dtype).eval()
        loaded = _Loaded(model=model, tok=tok, device=resolved)
        self._loaded = loaded
        return loaded

    # embedding source

    def embed(self, seqs: list[str]) -> dict[str, np.ndarray]:
        if not seqs:
            return {}
        loaded = self._load()

        clean = [sanitize_sequence(s) for s in seqs]
        max_pos = getattr(loaded.model.config, "max_position_embeddings", None)
        if max_pos is not None:
            for s in clean:
                if len(s) + 2 > max_pos:  # +2 for BOS/EOS tokens
                    raise SequenceTooLong(
                        f"sequence of length {len(s)} exceeds {self.model_name} "
                        f"capacity ({max_pos - 2} residues). Truncating would violate the "
                        f"alignment clause (L == len(seq), always) --- chunk externally or "
                        f"pick a longer-context model."
                    )

        batches = _plan_batches([len(s) for s in clean], MAX_TOKENS_PER_BATCH)
        need_hidden = self.layer is not None

        result: dict[str, np.ndarray] = {}
        for idxs in batches:
            enc = loaded.tok(
                [clean[i] for i in idxs],
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
            )
            input_ids = enc["input_ids"].to(loaded.device)
            attention = enc["attention_mask"].to(loaded.device)
            with torch.inference_mode():
                out = loaded.model(
                    input_ids=input_ids,
                    attention_mask=attention,
                    output_hidden_states=need_hidden,
                )
            hidden = out.hidden_states[self.layer] if need_hidden else out.last_hidden_state
            for row, i in enumerate(idxs):
                vec = (
                    hidden[row, 1 : 1 + len(clean[i]), :]
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )
                result[seqs[i]] = vec  # keyed to the caller's original string
        return result


class SequenceTooLong(ValueError):
    """Raised instead of truncating: silent truncation would violate the
    alignment clause (L == len(seq)) that the whole harness trusts."""
