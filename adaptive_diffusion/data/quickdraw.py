"""
Dibujos de trazo fino de Quick, Draw! renderizados en binario a 64x64.

Los dibujos vienen como trazos vectoriales (coordenadas 0..255). Aquí se pintan con
línea de 1 píxel: casi todo el dibujo es borde, que es el caso difícil para un quadtree.

La primera vez descarga las primeras líneas de cada categoría (sólo los dibujos que
el juego dio por reconocidos) y guarda el resultado en data/quickdraw/quickdraw<size>.npz.
"""
import json
import os
import subprocess

import numpy as np

CLASSES = ["airplane", "bicycle", "cat", "clock", "fish", "flower", "house", "star", "tree", "umbrella"]
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quickdraw")
_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{}.ndjson"


def render(strokes, size=64, margin=2):
    from PIL import Image, ImageDraw
    img = Image.new("1", (size, size), 0)
    d = ImageDraw.Draw(img)
    k = (size - 1 - 2 * margin) / 255.0
    for xs, ys in strokes:
        pts = [(margin + x * k, margin + y * k) for x, y in zip(xs, ys)]
        if len(pts) == 1:
            d.point(pts, 1)
        else:
            d.line(pts, fill=1, width=1)
    return np.asarray(img, dtype=np.uint8)


def _fetch(name, n):
    """Primeros n dibujos reconocidos de una categoría (lee sólo el principio del fichero)."""
    raw = os.path.join(_ROOT, "raw", f"{name}.ndjson")
    if not os.path.exists(raw):
        os.makedirs(os.path.dirname(raw), exist_ok=True)
        cmd = f"curl -sL --fail '{_URL.format(name.replace(' ', '%20'))}' | head -n {int(n * 1.25)} > '{raw}'"
        subprocess.run(["bash", "-c", cmd], check=False)
    out = []
    with open(raw, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("recognized"):
                out.append(d["drawing"])
                if len(out) == n:
                    break
    if len(out) < n:
        raise RuntimeError(f"{name}: sólo {len(out)} dibujos reconocidos de {n}")
    return out


def build(size=64, per_class_train=6000, per_class_test=1000):
    n = per_class_train + per_class_test
    imgs, labels = [], []
    for c, name in enumerate(CLASSES):
        print(f"quickdraw: {name}", flush=True)
        imgs.append(np.stack([render(s, size) for s in _fetch(name, n)]))
        labels.append(np.full(n, c, dtype=np.int64))
    imgs, labels = np.stack(imgs), np.stack(labels)           # (clases, n, S, S)
    tr = (imgs[:, :per_class_train].reshape(-1, size, size), labels[:, :per_class_train].reshape(-1))
    te = (imgs[:, per_class_train:].reshape(-1, size, size), labels[:, per_class_train:].reshape(-1))
    perm = np.random.default_rng(0).permutation(len(tr[0]))   # barajar una vez el entrenamiento
    path = os.path.join(_ROOT, f"quickdraw{size}.npz")
    np.savez_compressed(path, train_x=np.packbits(tr[0][perm], axis=-1), train_y=tr[1][perm],
                        test_x=np.packbits(te[0], axis=-1), test_y=te[1], size=size)
    return path


def load_quickdraw(train=True, size=64):
    """Devuelve (imgs, labels): imgs (N, size, size) uint8 en {0,1}."""
    path = os.path.join(_ROOT, f"quickdraw{size}.npz")
    if not os.path.exists(path):
        build(size)
    d = np.load(path)
    k = "train" if train else "test"
    return np.unpackbits(d[f"{k}_x"], axis=-1)[..., :size].astype(np.uint8), d[f"{k}_y"]


if __name__ == "__main__":
    x, y = load_quickdraw(True)
    print(x.shape, y.shape, "píxeles llenos %.1f%%" % (100 * x.mean()))
