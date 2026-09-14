from typing import Protocol

import numpy as np


class EmbeddingSource(Protocol):
    """
    Anything that turns protein sequences into per-residue embeddings.

    Third-party implementations must honor:

        - ALIGNMENT: ``embed(seqs)[seq]`` has shape ``(L, D)`` with
          ``L == len(seq)``, always. BOS/EOS and all special tokens are
          stripped; exactly one vector per residue, aligned 1:1 with the
          input string. X/B/U/Z each count as one residue.
        - DTYPE: float16.
        - DETERMINISM: identical input produces bitwise-identical output.
        - CACHE_ID: a stable string uniquely identifying (model, layer,
          weights version). The cache keys on it, so mutating a model's
          weights in place MUST change its cache_id. Without this member,
          two different models would silently share cache entries.

        Implementations do NOT inherit from this class
    """

    @property
    def dim(self) -> int: ...

    @property
    def cache_id(self) -> str: ...

    def embed(self, seqs: list[str]) -> dict[str, np.ndarray]: ...
