"""Piezas compartidas por los scripts 3D: datos de ABO filtrados, creación y carga del modelo."""
import os

import numpy as np
import torch

from data.abo import load_abo
from models.box_grow import BoxDataset, tile_token_count
from models.box_transformer import BoxTransformer

TREE_FLAGS = {"tile": dict(predict_cut=False, predict_full=False), "oct": dict(predict_cut=False), "omni": dict()}
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "abo", "cache")


def load_shapes(res, train, max_tile_tokens=None, limit=None):
    """Rejillas y etiquetas de ABO. El filtro de tamaño usa siempre el octree `tile`, para que todos los
    árboles se entrenen y evalúen con exactamente las mismas formas."""
    vox, labels = load_abo(res, train)
    if max_tile_tokens:
        keep = tile_token_count(vox) <= max_tile_tokens
        vox, labels = vox[keep], labels[keep]
    if limit:
        vox, labels = vox[:limit], labels[:limit]
    return vox, labels


def make_dataset(tree, res, train, aug, max_tile_tokens=None, limit=None, log=print, workers=0):
    vox, labels = load_shapes(res, train, max_tile_tokens, limit)
    name = f"{tree}_{res}_{'train' if train else 'test'}_a{aug}_m{max_tile_tokens}_n{len(vox)}.npz"
    return BoxDataset(vox, labels, tree, aug=aug, cache=os.path.join(CACHE, name), log=log, workers=workers)


def make_model(tree, depth, dim, layers, heads):
    return BoxTransformer(depth=depth, dim=dim, layers=layers, heads=heads, **TREE_FLAGS[tree])


def load_model(path, device):
    ck = torch.load(path, map_location=device)
    a = ck["args"]
    model = make_model(a["tree"], ck["depth"], a["dim"], a["layers"], a["heads"]).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), a
