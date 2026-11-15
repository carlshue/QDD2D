import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
torch = pytest.importorskip("torch")

from models.box_grow import BoxDataset, BucketSampler, augment, make_training_batch, sample, tile_token_count
from models.box_transformer import CUT_MASK, BoxTransformer
from structures.boxtree import C_CUT, C_DEPTH, build_tokens, check_tree, tokens_to_voxels
from tests.test_boxtree import _shapes

FLAGS = {"tile": dict(predict_cut=False, predict_full=False), "oct": dict(predict_cut=False),
         "omni": dict()}


def _data(kind, n=24, aug=1):
    v = _shapes(n, S=8)
    return v, BoxDataset(v, np.arange(n) % 10, kind, aug=aug, log=None)


def test_dataset_matches_builder_and_augmentations_are_lossless():
    v, data = _data("omni", n=8, aug=8)
    assert len(data) == 64
    ref = build_tokens(v, "omni")
    assert all((data[i][0].numpy() == ref[i]).all() for i in range(8))
    for k in range(8):                                  # cada variante decodifica a la rejilla girada
        want = augment(v, k)
        for i in range(8):
            assert (tokens_to_voxels(data[8 * k + i][0].numpy(), 3) == want[i]).all()
    assert (tile_token_count(v) == [len(t) for t in build_tokens(v, "tile")]).all()


def test_training_batch_masks_only_the_deepest_level_present():
    torch.manual_seed(0)
    _, data = _data("omni")
    b, labels = make_training_batch([data[i] for i in range(16)])
    assert b["masked"].any(1).all() and len(labels) == 16
    assert (b["cut"][b["masked"]] == CUT_MASK).all()
    for i in range(16):
        d = b["depth"][i][~b["pad"][i]]
        assert (b["depth"][i][b["masked"][i]] == d.max()).all()


def test_bucket_sampler_covers_the_data_in_full_batches():
    lengths = np.random.default_rng(0).integers(1, 100, 103)
    batches = list(BucketSampler(lengths, 8, pool=3))
    flat = [i for b in batches for i in b]
    assert all(len(b) == 8 for b in batches) and len(set(flat)) == len(flat) >= 96


@pytest.mark.parametrize("kind", ["tile", "oct", "omni"])
def test_loss_decreases_and_samples_are_valid_trees(kind):
    torch.manual_seed(0)
    _, data = _data(kind)
    model = BoxTransformer(depth=3, dim=48, layers=2, heads=2, **FLAGS[kind])
    opt = torch.optim.Adam(model.parameters(), 3e-3)
    losses = []
    for _ in range(80):
        b, labels = make_training_batch([data[i] for i in torch.randint(0, len(data), (16,)).tolist()])
        loss, parts = model.loss(b, labels)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert np.isfinite(losses).all() and np.mean(losses[-10:]) < np.mean(losses[:10])
    assert ("corte" in parts) == (kind == "omni")
    model.eval()
    trees, stats = sample(model, 6, steps_per_level=3, label=torch.arange(6))
    assert stats["truncados"] == 0
    for t in trees:
        assert not (t[:, C_CUT] == CUT_MASK).any()
        assert check_tree(t, 3) == [], kind
        assert tokens_to_voxels(t, 3).shape == (8, 8, 8)
        if kind != "omni":
            assert (t[:, C_CUT] == 7).all()
    capped, stats = sample(model, 4, steps_per_level=2, max_tokens=3)
    assert all(len(t) <= 3 for t in capped)
