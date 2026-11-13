import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
torch = pytest.importorskip("torch")

from models.adaptive_diffusion import loss_fn, make_training_batch, sample
from models.quadtree_transformer import MASK, QuadtreeTransformer
from structures.binary_quadtree import SPLIT, build_levels, levels_to_image, to_tokens, tokens_to_levels


def _trees(n=8, depth=3, seed=0):
    rng = np.random.default_rng(seed)
    imgs = (rng.random((n, 2 ** depth, 2 ** depth)) < 0.3).astype(np.uint8)
    levels = build_levels(imgs)
    return [torch.from_numpy(to_tokens([lv[i] for lv in levels])) for i in range(n)]


def test_training_batch_masks_only_target_level():
    torch.manual_seed(0)
    batch = make_training_batch(_trees(), depth=3)
    masked = batch["state"] == MASK
    assert masked.any(1).all()
    assert ((batch["target"] != -100) == masked).all()
    for i in range(len(masked)):       # todo lo enmascarado está en el nivel más fino de la secuencia
        lv = batch["level"][i][~batch["pad_mask"][i]]
        assert (batch["level"][i][masked[i]] == lv.max()).all()


def test_loss_decreases_and_samples_are_valid_trees():
    torch.manual_seed(0)
    trees = _trees(16)
    model = QuadtreeTransformer(depth=3, dim=32, layers=2, heads=2)
    opt = torch.optim.Adam(model.parameters(), 3e-3)
    losses = []
    for _ in range(60):
        loss = loss_fn(model, make_training_batch(trees, 3))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert np.mean(losses[-10:]) < np.mean(losses[:10])
    model.eval()
    out, _, _ = sample(model, 4, steps_per_level=3)
    for t in out:
        assert not (t[:, 3] == MASK).any()
        assert not ((t[:, 0] == 3) & (t[:, 3] == SPLIT)).any()
        levels = tokens_to_levels(t.numpy(), 3)
        img = levels_to_image([lv[None] for lv in levels])
        # el árbol generado decodifica a una imagen cuyo árbol tiene como mucho esos nodos
        assert len(to_tokens([lv[0] for lv in build_levels(img)])) <= len(t)


def _valid(t, depth=3):
    assert not (t[:, 3] == MASK).any()
    assert not ((t[:, 0] == depth) & (t[:, 3] == SPLIT)).any()
    for l in range(depth):       # cada SPLIT tiene exactamente sus 4 hijos y no hay huérfanos
        assert ((t[:, 0] == l + 1).sum() == 4 * ((t[:, 0] == l) & (t[:, 3] == SPLIT)).sum())


def test_remask_keeps_trees_valid_and_counts():
    torch.manual_seed(0)
    model = QuadtreeTransformer(depth=3, dim=32, layers=2, heads=2).eval()
    base = {}
    out, ev0, _ = sample(model, 6, steps_per_level=4, stats=base)
    assert base["reenmascarados"] == 0 and base["cambios"] == 0
    for mode in ("conf", "uniform"):
        for rounds in (0, 2):
            st = {}
            g = torch.Generator().manual_seed(1)
            out, ev, _ = sample(model, 6, steps_per_level=4, generator=g, remask=0.5, remask_rounds=rounds,
                                remask_mode=mode, stats=st)
            for t in out:
                _valid(t)
            assert st["reenmascarados"] > 0
            assert st["cambios"] <= st["reenmascarados"]
            assert st["uniones"] + st["subdivisiones_tardias"] <= st["cambios"]
            if rounds:
                assert ev > ev0


def test_remask_off_is_reproducible():
    model = QuadtreeTransformer(depth=3, dim=32, layers=2, heads=2).eval()
    a, _, _ = sample(model, 4, steps_per_level=3, generator=torch.Generator().manual_seed(5))
    b, _, _ = sample(model, 4, steps_per_level=3, generator=torch.Generator().manual_seed(5), remask=0.0, remask_rounds=3)
    assert all((x == y).all() for x, y in zip(a, b))
