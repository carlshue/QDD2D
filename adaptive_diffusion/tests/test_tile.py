import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
torch = pytest.importorskip("torch")

from models.quadtree_transformer import QuadtreeTransformer
from models.tile_grow import T_MASK, images_to_tokens, make_training_batch, sample, tokens_to_image


def test_roundtrip_is_lossless():
    rng = np.random.default_rng(0)
    imgs = (rng.random((20, 16, 16)) < 0.1).astype(np.uint8)
    imgs[0] = 0; imgs[1] = 1; imgs[2, :8, :8] = 1
    toks = images_to_tokens(imgs)
    for im, t in zip(imgs, toks):
        assert (tokens_to_image(t, 4).numpy() == im).all()
        for l in range(3):       # un hijo por cada bit a 1 del nivel anterior
            par = t[t[:, 0] == l, 3]
            assert sum(bin(int(c)).count("1") for c in par) == int((t[:, 0] == l + 1).sum())
    assert len(toks[0]) == 1 and int(toks[0][0, 3]) == 0


def test_batch_and_sample():
    torch.manual_seed(0)
    imgs = (np.random.default_rng(1).random((8, 16, 16)) < 0.1).astype(np.uint8)
    b = make_training_batch(images_to_tokens(imgs), 4)
    assert ((b["target"] != -100) == (b["state"] == T_MASK)).all()
    model = QuadtreeTransformer(depth=4, dim=32, layers=2, heads=2, num_states=16).eval()
    trees, evals, tok = sample(model, 4, steps_per_level=3)
    for t in trees:
        assert not (t[:, 3] == T_MASK).any() and int(t[:, 0].max()) <= 3
        assert tokens_to_image(t, 4).shape == (16, 16)
