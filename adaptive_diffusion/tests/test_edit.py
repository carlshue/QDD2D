import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
torch = pytest.importorskip("torch")

from models.edit_diffusion import loss_fn, sample, training_batch
from models.quadtree_transformer import QuadtreeTransformer
from structures.binary_quadtree import ABSENT, SPLIT, build_levels
from structures.pyramid import Pyramid


def _data(n=32, depth=3, seed=0):
    rng = np.random.default_rng(seed)
    imgs = (rng.random((n, 2 ** depth, 2 ** depth)) < 0.3).astype(np.uint8)
    imgs[: n // 2, :, : 2 ** depth // 2] = 0     # zonas uniformes, para que haya hojas grandes
    return imgs


def test_pyramid_matches_numpy_quadtree_and_roundtrips():
    imgs = _data()
    pyr = Pyramid(3)
    s0 = pyr.from_images(torch.from_numpy(imgs))
    ex = pyr.exists(s0)
    levels = build_levels(imgs)
    for l, lv in enumerate(levels):
        sl = slice(pyr.offsets[l], pyr.offsets[l + 1])
        ref = torch.from_numpy(lv.reshape(len(imgs), -1))
        assert (ex[:, sl] == (ref != ABSENT)).all()
        assert (s0[:, sl][ex[:, sl]] == ref[ex[:, sl]]).all()
    assert (pyr.to_image(s0, ex).numpy() == imgs.astype(bool)).all()


def test_corruption_limits_and_priors():
    torch.manual_seed(0)
    pyr = Pyramid(3)
    s0 = pyr.from_images(torch.from_numpy(_data()))
    for prior in (pyr.prior_uniform(0.6), pyr.prior_from_data(s0, "level"), pyr.prior_from_data(s0, "cell")):
        assert torch.allclose(prior.sum(-1), torch.ones(pyr.size))
        assert (prior[pyr.offsets[3]:, SPLIT] == 0).all()        # un píxel no se subdivide
        clean, ex = pyr.corrupt(s0, torch.zeros(len(s0)), prior)
        assert (clean == s0).all() and (ex == pyr.exists(s0)).all()
        noisy, ex = pyr.corrupt(s0, torch.ones(len(s0)), prior)
        assert ex[:, 0].all() and not (noisy[:, pyr.offsets[3]:] == SPLIT).any()
    # el ruido por celda reproduce el tamaño medio de los árboles reales
    cell = pyr.prior_from_data(s0, "cell", smoothing=0.0)
    big = pyr.exists(pyr.sample_states(cell, 20000)).sum(1).float().mean()
    assert abs(big - pyr.exists(s0).sum(1).float().mean()) < 1.0


def test_training_and_sampling():
    torch.manual_seed(0)
    pyr = Pyramid(3)
    s0 = pyr.from_images(torch.from_numpy(_data(64)))
    prior = pyr.prior_from_data(s0, "cell")
    model = QuadtreeTransformer(depth=3, dim=32, layers=2, heads=2, use_time=True)
    opt = torch.optim.Adam(model.parameters(), 3e-3)
    losses = []
    for _ in range(80):
        batch = training_batch(pyr, s0, prior)
        assert ((batch["target"] == -100) == batch["pad_mask"]).all()
        loss, _ = loss_fn(model, batch)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert np.mean(losses[-10:]) < np.mean(losses[:10])
    model.eval()
    imgs, stats = sample(model, pyr, prior, 8, steps=6)
    assert imgs.shape == (8, 8, 8) and imgs.dtype == torch.bool
    assert stats["sin_resolver"] == 0
