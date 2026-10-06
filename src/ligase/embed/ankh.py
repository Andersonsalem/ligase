from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ligase.embed.batching import MAX_TOKENS_PER_BATCH, plan_batches
from ligase.embed.esm import sanitize_sequence

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "ElnaggarLab/ankh-base"
INPUT_CONVENTION = "raw"


@dataclass(frozen=True)
class _Loaded:
    """Everything embed needs once the weights are loaded on the device"""

    model: Any  # T5EncoderModel
    tok: Any  # tokenizer with padding_side forced to right
    device: torch.device
    n_prefix: int  # special token prepended to every sequence
    n_suffix: int  # special tokens appended to every sequence
    n_specials: int  # n_pref + n_suf (needed for alignment check)


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
        self._cfg: Any | None = None
        self._loaded: _Loaded | None = None

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

    def _load(self) -> _Loaded:
        if self._loaded is not None:
            return self._loaded
        from transformers import AutoTokenizer, T5EncoderModel

        config = self.hf_config
        n = getattr(config, "num_layers", None)
        if self.layer is not None and n is not None and not 0 <= self.layer <= n:
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
        tok.padding_side = "right"  # slice offsets assume right padding
        model = (
            T5EncoderModel.from_pretrained(self.model_name)  # type: ignore[call-arg]
            .to(device=resolved, dtype=dtype)
            .eval()
        )

        # load-time layout oracle (tokenizer only)
        specials = set(tok.all_special_ids)
        if tok.unk_token_id is not None:
            specials.add(tok.unk_token_id)  # probed by yours truly
        dummy = "AAA"
        ids = tok(dummy, add_special_tokens=True)["input_ids"]
        n_prefix, n_suffix = resolve_token_layout(ids, specials)
        mid = ids[n_prefix : len(ids) - n_suffix]
        pieces = tok.convert_ids_to_tokens(mid)
        if len(mid) != len(dummy) or pieces != list(dummy):
            raise ValueError(
                f"unexpected tokenizer layout for {self.model_name}: ids={ids}, resolved "
                f"(prefix={n_prefix}, suffix={n_suffix}), middle tokens={pieces} "
                "--- expected exactly one token per residue; paste this output"
            )
        logger.info(
            "token layout: %d prefix + %d suffix specials (input convention: %s)",
            n_prefix,
            n_suffix,
            INPUT_CONVENTION,
        )
        loaded = _Loaded(
            model=model,
            tok=tok,
            device=resolved,
            n_prefix=n_prefix,
            n_suffix=n_suffix,
            n_specials=n_prefix + n_suffix,
        )
        self._loaded = loaded
        return loaded

    def embed(self, seqs: list[str]) -> dict[str, np.ndarray]:
        if not seqs:
            return {}
        loaded = self._load()

        clean = [sanitize_sequence(s) for s in seqs]
        lengths = [len(s) for s in clean]
        batches = plan_batches(lengths, MAX_TOKENS_PER_BATCH)
        need_hidden = self.layer is not None

        result: dict[str, np.ndarray] = {}
        for idxs in batches:
            enc = loaded.tok(
                [clean[i] for i in idxs],  # RAW — see module docstring
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
                L = lengths[i]
                non_pad = int(attention[row].sum())
                if non_pad != L + loaded.n_specials:
                    raise ValueError(
                        f"sequence {i}: tokenizer produced {non_pad - loaded.n_specials} "
                        f"residue tokens for L={L} --- alignment clause violated "
                        f"(a residue split into multiple tokens or dropped); "
                        f"refusing to embed"
                    )
                vec = (
                    hidden[row, loaded.n_prefix : loaded.n_prefix + L, :]
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )
                result[seqs[i]] = vec  # keyed to the caller's original string
        return result
