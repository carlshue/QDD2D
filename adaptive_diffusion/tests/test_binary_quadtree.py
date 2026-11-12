import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from structures.binary_quadtree import (ABSENT, EMPTY, FULL, SPLIT, build_levels, levels_to_image,
                                        node_counts, to_tokens, tokens_to_levels)


def test_roundtrip_random():
    rng = np.random.default_rng(0)
    for p in (0.0, 0.05, 0.5, 1.0):
        imgs = (rng.random((20, 32, 32)) < p).astype(np.uint8)
        assert (levels_to_image(build_levels(imgs)) == imgs).all()


def test_uniform_images_are_single_node():
    imgs = np.stack([np.zeros((8, 8)), np.ones((8, 8))]).astype(np.uint8)
    levels = build_levels(imgs)
    assert levels[0][:, 0, 0].tolist() == [EMPTY, FULL]
    assert node_counts(levels).sum(1).tolist() == [1, 1]


def test_structure_is_consistent():
    img = np.zeros((1, 8, 8), dtype=np.uint8)
    img[0, 0, 0] = 1
    levels = build_levels(img)
    assert node_counts(levels)[0].tolist() == [1, 4, 4, 4]
    for l in range(len(levels) - 1):   # un hijo existe si y sólo si su padre es SPLIT
        parent = np.repeat(np.repeat(levels[l] == SPLIT, 2, 1), 2, 2)
        assert ((levels[l + 1] != ABSENT) == parent).all()
    assert not (levels[-1] == SPLIT).any()


def test_tokens_roundtrip_and_morton_groups_siblings():
    rng = np.random.default_rng(1)
    img = (rng.random((1, 16, 16)) < 0.3).astype(np.uint8)
    one = [lv[0] for lv in build_levels(img)]
    tok = to_tokens(one)
    back = tokens_to_levels(tok, 4)
    assert all((a == b).all() for a, b in zip(one, back))
    for l in range(1, 5):              # los 4 hermanos van seguidos
        t = tok[tok[:, 0] == l]
        parents = (t[:, 1] // 2) * 1000 + t[:, 2] // 2
        assert (parents.reshape(-1, 4) == parents.reshape(-1, 4)[:, :1]).all()
