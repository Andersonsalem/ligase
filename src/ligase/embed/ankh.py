from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ligase.embed.batching import MAX_TOKENS_PER_BATCH, plan_batches
from ligase.embed.esm import sanitize_sequence

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "ElnaggarLab/ankh-base"
INPUT_CONVENTION = "raw"


def resolve_token_layout(ids: list[int], special_ids: set[int]) -> tuple[int, int]:
    """
    Count leading and trailing tokens in a tokenized dummy sequence

    Returns (n_prefix, n_suffix), raises if specials appear mid-seq
    """
    n = len(ids)
    flags = [i in special_ids for i in ids]
    pre = 0
    while pre < n and flags[pre]:
        pre += 1
    suf = 0
    while suf < n - pre and flags[n - 1 - suf]:
        suf += 1
    if any(flags[pre : n - suf]):
        raise ValueError(f"special tokens appear mid-sequence: {ids}")
    return pre, suf


class AnkhSource:
    """
    Ankh T5-encoder checkpoint satisfying EmbeddingSource protocol
    Loaded lazily
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        layer: int | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.layer = layer
        self.device = device
        self._dim: int | None = None
        self._tok: Any | None = None
        self._model: Any | None = None
        self._cfg: Any | None = None
        self._resolved_device: Any | None = None
        self._torch: Any | None = None
        self._n_prefix = 0
        self._n_suffix = 0
        self._n_specials = 0

    @property
    def cache_id(self) -> str:
        return (
            f"ankh|{self.model_name}|layer={self.layer if self.layer is not None else 'last'}"
            f"|{INPUT_CONVENTION}"
        )

    @property
    def hf_config(self):
        if self._cfg is None:
            from transformers import AutoConfig

            self._cfg = AutoConfig.from_pretrained(self.model_name)
        return self._cfg

    @property
    def dim(self) -> int:
        return int(getattr(self.hf_config, "d_model", None) or self.hf_config.hidden_size)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, T5EncoderModel

        config = self.hf_config
        n = getattr(config, "num_layers", None)
        if self.layer is not None and n is not None and not 0 <= self.layer <= n:
            raise ValueError(
                f"ankh|{self.model_name}|layer={self.layer if self.layer is not None else 'last'}"
                f"|{INPUT_CONVENTION}"
            )
        resolved = (
            torch.device(self.device)
            if self.device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        dtype = torch.bfloat16 if resolved.type == "cuda" else torch.float32
        logger.info("loading %s on %s (%s)", self.model_name, resolved, dtype)
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._tok.padding_side = "right"  # slice offsets assume right padding
        self._model = (
            T5EncoderModel.from_pretrained(self.model_name).to(device=resolved, dtype=dtype).eval()
        )
        self._resolved_device = resolved
        self._torch = torch

        # load time layout oracle (tokenizer only; precedes any heavy compute)
        specials = set(self._tok.all_special_ids)
        if self._tok.unk_token_id is not None:
            specials.add(self._tok.unk_token_id)  # probed by yours truly
        dummy = "AAA"
        ids = self._tok(dummy, add_special_tokens=True)["input_ids"]
        self._n_prefix, self._n_suffix = resolve_token_layout(ids, specials)
        mid = ids[self._n_prefix : len(ids) - self._n_suffix]
        pieces = self._tok.convert_ids_to_tokens(mid)
        if len(mid) != len(dummy) or pieces != list(dummy):
            raise ValueError(
                f"unexpected tokenizer layout for {self.model_name}: ids={ids}, resolved "
                f"(prefix={self._n_prefix}, suffix={self._n_suffix}), middle tokens={pieces} "
                "--- expected exactly one token per residue; paste this output"
            )
        self._n_specials = self._n_prefix + self._n_suffix
        logger.info(
            "token layout: %d prefix + %d suffix specials (input convention: %s)",
            self._n_prefix,
            self._n_suffix,
            INPUT_CONVENTION,
        )

    def embed(self, seqs: list[str]) -> dict[str, np.ndarray]:
        if not seqs:
            return {}
        self._ensure_loaded()
        torch = self._torch

        clean = [sanitize_sequence(s) for s in seqs]
        lengths = [len(s) for s in clean]
        batches = plan_batches(lengths, MAX_TOKENS_PER_BATCH)
        need_hidden = self.layer is not None

        result: dict[str, np.ndarray] = {}
        for idxs in batches:
            enc = self._tok(
                [clean[i] for i in idxs],  # RAW — see module docstring
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
            )
            input_ids = enc["input_ids"].to(self._resolved_device)
            attention = enc["attention_mask"].to(self._resolved_device)
            with torch.inference_mode():
                out = self._model(
                    input_ids=input_ids,
                    attention_mask=attention,
                    output_hidden_states=need_hidden,
                )
            hidden = out.hidden_states[self.layer] if need_hidden else out.last_hidden_state
            for row, i in enumerate(idxs):
                L = lengths[i]
                non_pad = int(attention[row].sum())
                if non_pad != L + self._n_specials:
                    raise ValueError(
                        f"sequence {i}: tokenizer produced {non_pad - self._n_specials} "
                        f"residue tokens for L={L} --- alignment clause violated "
                        f"(a residue split into multiple tokens or dropped); "
                        f"refusing to embed"
                    )
                vec = (
                    hidden[row, self._n_prefix : self._n_prefix + L, :]
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )
                result[seqs[i]] = vec  # keyed to the caller's original string
        return result
