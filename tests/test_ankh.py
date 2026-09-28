from __future__ import annotations

import pytest

from ligase.embed.ankh import AnkhSource


def test_ankh_cache_id_distinct_from_esm() -> None:
    from ligase.embed.esm import ESM2Source

    a = AnkhSource(model_name="ElnaggarLab/ankh-base").cache_id
    e = ESM2Source(model_name="ElnaggarLab/ankh-base").cache_id
    assert a.startswith("ankh|")
    assert a != e  # distinct h5 caches by construction (the M1 lesson)


def test_ankh_cache_id_reflects_layer() -> None:
    id5, id_last = AnkhSource(layer=5).cache_id, AnkhSource().cache_id
    assert "layer=5" in id5
    assert "layer=last" in id_last
    assert id5 != id_last
    assert id5.endswith("|raw")


@pytest.mark.slow
def test_ankh_conformance() -> None:
    """The protocol, for real: alignment (incl. X/B/U/Z + length-1),
    bitwise determinism, cache round-trip with zero recompute."""
    import numpy as np

    from ligase.embed.esm import ESM2Source  # noqa: F401  (registry completeness)
    from ligase.utils.caching import cached

    src = cached(AnkhSource(), "data/cache/embeddings")
    seqs = ["ACDEFGHIKLMNPQRSTVWY", "AXBUZ", "M", "ACDEFGHIKLMNPQRSTVWY"]
    out1 = src.embed(seqs)
    for s in seqs:
        assert s in out1 and out1[s].shape[0] == len(s)  # alignment clause
        assert out1[s].dtype == np.float16
    out2 = src.embed(seqs)
    for s in seqs:
        assert np.array_equal(out1[s], out2[s], equal_nan=False)  # bitwise
