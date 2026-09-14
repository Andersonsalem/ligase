from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

if TYPE_CHECKING:
    from ligase.embed.protocol import EmbeddingSource

logger = logging.getLogger(__name__)


def _seq_key(cache_id: str, seq: str) -> str:
    return hashlib.sha256(f"{cache_id}|{seq}".encode()).hexdigest()


def _file_name(cache_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9]", "_", cache_id)[:48]
    digest = hashlib.sha256(cache_id.encode()).hexdigest()[:12]
    return f"{readable}-{digest}.h5"


class _CachedSource:
    """Protocol preserving wrapper: same dim, cache_id and cached embed()"""

    def __init__(self, inner: EmbeddingSource, cache_dir: Path):
        self._inner = inner
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._path = cache_dir / _file_name(inner.cache_id)

    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def cache_id(self) -> str:
        return self._inner.cache_id

    def embed(self, seqs: list[str]) -> dict[str, np.ndarray]:
        unique = list(dict.fromkeys(seqs))  # deduplicate and preserve order
        found = self._load(unique)
        missing = [s for s in unique if s not in found]
        if missing:
            fresh = self._inner.embed(missing)
            self._store(fresh)
            found.update(fresh)
        return {s: found[s] for s in seqs}

    def _load(self, seqs: list[str]) -> dict[str, np.ndarray]:
        if not self._path.exists():
            return {}
        out: dict[str, np.ndarray] = {}
        try:
            with h5py.File(self._path, "r") as f:
                for seq in seqs:
                    key = _seq_key(self._inner.cache_id, seq)
                    if key in f:
                        try:
                            out[seq] = np.asarray(f[key][()], dtype=np.float16)
                        except (OSError, KeyError, TypeError):
                            logger.warning(f"torn cache entry for sequence {seq} - recomputing")
        except OSError as e:
            logger.warning("cache unreadable (%s) - recomputing everything", e)
        return out

    def _store(self, fresh: dict[str, np.ndarray]) -> None:
        try:
            with h5py.File(self._path, "a") as f:
                for seq, arr in fresh.items():
                    key = _seq_key(self._inner.cache_id, seq)
                    if key not in f:
                        f.create_dataset(key, data=arr)
        except OSError as e:
            # caching is just for optimization, a run shouldn't really fail over it
            logger.warning("cache unwriteable (%s) - skipping", e)


def cached(
    source: EmbeddingSource, cache_dir: Path = Path("data/cache/embeddings")
) -> EmbeddingSource:
    """
    Wrap any EmbeddingSource with the hash-addressed disk cache.
    """
    return _CachedSource(source, cache_dir)  # type: ignore[return-value]
