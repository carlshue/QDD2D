"""
MNIST binarizado y rellenado a 32x32, leído directamente de los ficheros IDX
(sin dependencia de torchvision).
"""
import gzip
import os
import struct
import numpy as np

_DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MNIST", "raw")


def _read_idx(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        _, _, dims = struct.unpack(">HBB", f.read(4))
        shape = struct.unpack(">" + "I" * dims, f.read(4 * dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def _find(root, name):
    for cand in (os.path.join(root, name), os.path.join(root, name + ".gz")):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"No encuentro {name} en {root}")


def load_binary_mnist(train=True, root=_DEFAULT_ROOT, threshold=127, size=32):
    """Devuelve (imgs, labels): imgs (N, size, size) uint8 en {0,1}, dígito centrado."""
    prefix = "train" if train else "t10k"
    imgs = _read_idx(_find(root, f"{prefix}-images-idx3-ubyte"))
    labels = _read_idx(_find(root, f"{prefix}-labels-idx1-ubyte"))
    binary = (imgs > threshold).astype(np.uint8)
    pad = (size - binary.shape[1]) // 2
    binary = np.pad(binary, ((0, 0), (pad, pad), (pad, pad)))
    return binary, labels.astype(np.int64)
