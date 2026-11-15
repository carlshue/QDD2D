import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
torch = pytest.importorskip("torch")

from models.quadtree_transformer import QuadtreeTransformer
from models.tile_grow import T_MASK, T_PAD, images_to_tokens
from models.tile_joint import CachedNet, Grid, loss_fn, make_training_batch, sample


def _imgs(n=12, s=16, seed=0):
    return torch.from_numpy((np.random.default_rng(seed).random((n, s, s)) < 0.08).astype(np.uint8))


def test_dense_matches_token_list_and_roundtrip():
    imgs = _imgs()
    g = Grid(4)
    val, ex = g.from_images(imgs)
    assert (g.existence(val) == ex).all()
    assert (g.to_image(val, ex) == imgs).all()
    for i, t in enumerate(images_to_tokens(imgs.numpy())):
        assert int(ex[i].sum()) == len(t)
        flat = [g.slices[int(l)].start + int(y) * 2 ** int(l) + int(x) for l, y, x, _ in t]
        assert (val[i, flat] == t[:, 3]).all() and ex[i, flat].all()


def test_training_batch_hides_descendants_of_masked():
    torch.manual_seed(0)
    g = Grid(4)
    b = make_training_batch(g, _imgs(32))
    assert ((b["target"] != -100) == (b["state"] == T_MASK)).all()
    assert (b["state"] == T_MASK).any(1).all()
    for i in range(32):          # ninguna celda presente cuelga de una celda tapada
        on = ~b["pad_mask"][i]
        cells = {(int(l), int(y), int(x)): int(s) for l, y, x, s in
                 zip(b["level"][i][on], b["y"][i][on], b["x"][i][on], b["state"][i][on])}
        for (l, y, x) in cells:
            if l > 0:
                par = cells[(l - 1, y // 2, x // 2)]
                assert par != T_MASK and (par >> (2 * (y % 2) + x % 2)) & 1
    model = QuadtreeTransformer(depth=4, dim=32, layers=2, heads=2, num_states=16)
    assert torch.isfinite(loss_fn(model, b))


@pytest.mark.parametrize("local", [None, (1, 0), (2, 1)])
def test_cached_net_full_recompute_equals_forward(local):
    torch.manual_seed(0)
    g = Grid(4)
    model = QuadtreeTransformer(depth=4, dim=32, layers=2, heads=2, num_states=16, local=local).eval()
    val, ex = g.from_images(_imgs(6))
    val[:, g.slices[3]] = T_MASK
    lab = torch.arange(6) % 10
    dense = CachedNet(model, g, 6, lab)(val, ex, ex)
    order, pad = g.sequence(ex)
    st = torch.where(pad, torch.full_like(order, T_PAD), val.gather(1, order))
    ref = model(st, g.level[order], g.y[order], g.x[order], pad, lab)
    assert torch.allclose(dense.gather(1, order[..., None].expand_as(ref))[~pad], ref[~pad], atol=1e-4)


def test_sample_all_modes_give_complete_trees():
    model = QuadtreeTransformer(depth=4, dim=32, layers=2, heads=2, num_states=16).eval()
    base = None
    for remask in ("none", "oldest", "conf"):
        for heat in (None, 0, 2):
            g = torch.Generator().manual_seed(3)
            imgs, st = sample(model, 5, steps=8, generator=g, remask=remask, heat=heat)
            assert imgs.shape == (5, 16, 16) and imgs.max() <= 1
            assert st["tokens_por_muestra"] <= st["tokens_sin_cache"] + 1e-6
            if heat is None:
                assert st["tokens_por_muestra"] == st["tokens_sin_cache"]
            assert (st["reenmascarados"] == 0) == (remask == "none")


def test_local_attention_ignores_far_cells():
    torch.manual_seed(0)
    model = QuadtreeTransformer(depth=5, dim=32, layers=1, heads=2, num_states=16, local=(1, -1)).eval()
    lv = torch.tensor([[4, 4, 4, 3]]); y = torch.tensor([[0, 1, 9, 0]]); x = torch.tensor([[0, 1, 9, 0]])
    pad = torch.zeros(1, 4, dtype=torch.bool)
    a = model(torch.tensor([[1, 2, 3, 5]]), lv, y, x, pad)
    b = model(torch.tensor([[1, 2, 9, 5]]), lv, y, x, pad)       # cambia sólo la celda lejana
    assert torch.allclose(a[0, :2], b[0, :2], atol=1e-5) and torch.allclose(a[0, 3], b[0, 3], atol=1e-5)
    assert not torch.allclose(a[0, 2], b[0, 2], atol=1e-5)
