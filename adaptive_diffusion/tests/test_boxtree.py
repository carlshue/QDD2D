import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.omnitree_stats3d import analyse, tile_stats
from structures.boxtree import (C_CUT, C_DEPTH, C_FULL, C_OCC, build_tokens, check_tree, tokens_to_voxels)


def _shapes(n=24, S=16, seed=0):
    """Mezcla de ruido disperso, losas, cajas y una esfera: hay hojas grandes y cortes de un solo eje."""
    rng = np.random.default_rng(seed)
    v = (rng.random((n, S, S, S)) < 0.02).astype(np.uint8)
    for i in range(n):
        a, b = sorted(rng.integers(0, S, 2))
        if i % 4 == 0:
            v[i, a:b + 1, :, :] = 1                      # losa gruesa
        elif i % 4 == 1:
            v[i, :, rng.integers(0, S), :] = 1           # plano de un vóxel
        elif i % 4 == 2:
            c = rng.integers(2, S - 2, 3)
            v[i, c[0] - 2:c[0] + 3, c[1] - 1:c[1] + 2, c[2]:c[2] + 6] = 1
        else:
            g = np.indices((S, S, S)) - S / 2 + 0.5
            v[i] |= (np.abs(np.sqrt((g ** 2).sum(0)) - S / 3) < 0.8).astype(np.uint8)
    return v


def test_roundtrip_and_wellformed():
    v = _shapes()
    for kind in ("tile", "oct", "omni"):
        toks = build_tokens(v, kind)
        assert len(toks) == len(v)
        for t, ref in zip(toks, v):
            assert (np.diff(t[:, C_DEPTH]) >= 0).all()
            assert check_tree(t, 4) == [], kind
            assert (tokens_to_voxels(t, 4) == ref).all(), kind


def test_token_counts_match_the_measurement_script():
    v = _shapes(16)
    for kind in ("oct", "omni"):
        n_tok = np.array([len(t) for t in build_tokens(v, kind)])
        assert (n_tok == analyse(v, kind)["tokens"]).all()
    assert (np.array([len(t) for t in build_tokens(v, "tile")]) == tile_stats(v)["tokens"]).all()


def test_omnitree_is_never_larger_and_uses_single_axis_cuts():
    v = _shapes(16)
    omni, oct3 = build_tokens(v, "omni"), build_tokens(v, "oct")
    assert all(len(a) <= len(b) for a, b in zip(omni, oct3))
    cuts = np.concatenate([t[:, C_CUT] for t in omni])
    assert set(np.unique(cuts)) - {7}                         # hay cortes que no son de los tres ejes
    assert all((t[:, C_CUT] == 7).all() for t in oct3)


def test_tile_only_has_full_leaves_at_voxel_level():
    v = _shapes(8)
    for t in build_tokens(v, "tile"):
        coarse = t[t[:, C_DEPTH] < 3]
        assert (coarse[:, C_FULL] == 0).all()
        fine = t[t[:, C_DEPTH] == 3]
        assert (fine[:, C_FULL] == fine[:, C_OCC]).all()
