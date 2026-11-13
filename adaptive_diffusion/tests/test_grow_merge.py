import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
torch = pytest.importorskip("torch")

from models.edit_diffusion import sample as edit_sample
from models.grow_merge import loss_fn, sample, training_batch
from models.heat import HeatTrace
from models.quadtree_transformer import MASK, PAD, QuadtreeTransformer
from structures.binary_quadtree import SPLIT
from structures.pyramid import Pyramid


def _data(n=64, depth=3, seed=0):
    rng = np.random.default_rng(seed)
    imgs = (rng.random((n, 2 ** depth, 2 ** depth)) < 0.3).astype(np.uint8)
    imgs[: n // 2, :, : 2 ** depth // 2] = 0
    return imgs


def test_training_batch_is_a_valid_partial_tree():
    torch.manual_seed(0)
    pyr = Pyramid(3)
    s0 = pyr.from_images(torch.from_numpy(_data()))
    prior = pyr.prior_from_data(s0, "cell")
    b = training_batch(pyr, s0, prior, eps_max=0.3)
    assert ((b["target"] == -100) == b["pad_mask"]).all()
    real = ~b["pad_mask"]
    assert (b["level"][real].max() <= pyr.depth) and (b["state"][real] <= MASK).all()
    assert (b["state"][real] == MASK).any()
    for i in range(len(s0)):          # todo nodo no raíz cuelga de un padre presente y en SPLIT
        nodes = {(int(l), int(y), int(x)): int(s) for l, y, x, s, r in
                 zip(b["level"][i], b["y"][i], b["x"][i], b["state"][i], real[i]) if r}
        assert (0, 0, 0) in nodes
        for (l, y, x) in nodes:
            if l > 0:
                assert nodes.get((l - 1, y // 2, x // 2)) == SPLIT
    # sin estropear nada, lo no tapado coincide con la verdad
    clean = training_batch(pyr, s0, prior, eps_max=0.0)
    seen = ~clean["pad_mask"] & (clean["state"] != MASK)
    assert (clean["state"][seen] == clean["target"][seen]).all()


def test_training_and_sampling_with_and_without_merge():
    torch.manual_seed(0)
    pyr = Pyramid(3)
    s0 = pyr.from_images(torch.from_numpy(_data()))
    prior = pyr.prior_from_data(s0, "cell")
    model = QuadtreeTransformer(depth=3, dim=32, layers=2, heads=2, use_time=True)
    opt = torch.optim.Adam(model.parameters(), 3e-3)
    losses = []
    for _ in range(80):
        loss, _ = loss_fn(model, training_batch(pyr, s0, prior))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert np.mean(losses[-10:]) < np.mean(losses[:10])
    model.eval()
    for revise, thr in ((0.0, 0.5), (1.0, 0.5), (0.5, 1.0)):
        imgs, st = sample(model, pyr, 8, steps_per_level=4, revise=revise, revise_thr=thr)
        assert imgs.shape == (8, 8, 8) and imgs.dtype == torch.bool
        assert st["sin_resolver"] == 0
        if revise == 0:
            assert st["revisiones"] == 0 and st["uniones"] == 0
    # la traza de nodos calientes no altera las muestras y cuadra con las estadísticas
    for fn in (lambda tr: sample(model, pyr, 8, steps_per_level=4, trace=tr, generator=torch.Generator().manual_seed(1)),
               lambda tr: edit_sample(model, pyr, prior, 8, steps=6, trace=tr, generator=torch.Generator().manual_seed(1))):
        tr = HeatTrace(pyr, thr=0.9, window_levels=(1, 2))
        a, st = fn(tr)
        b, _ = fn(None)
        assert (a == b).all()
        s = tr.summary()
        assert s["evaluaciones"] == st["evals"] and abs(s["tokens_por_muestra"] - st["tokens_por_muestra"]) < 0.1
        assert 0 <= s["nodos_calientes_pct"] <= 100
        for wl in (1, 2):
            w = s[f"ventanas_n{wl}"]
            assert s["nodos_calientes_pct"] <= w["tokens_a_recalcular_pct"] <= 100
            assert w["cambios_fuera_de_ventanas_calientes_pct"] <= s["cambios_en_nodos_frios_pct"] + 1e-6
