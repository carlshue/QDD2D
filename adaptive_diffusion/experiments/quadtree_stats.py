"""
Fase 0: ¿cuántos nodos necesita el quadtree binario frente a los 1024 píxeles?
Uso (desde adaptive_diffusion/):  python -m experiments.quadtree_stats
"""
import numpy as np
from data.binary_mnist import load_binary_mnist
from structures.binary_quadtree import build_levels, levels_to_image, node_counts, SPLIT, ABSENT


def main():
    for split in (True, False):
        imgs, _ = load_binary_mnist(train=split)
        levels = build_levels(imgs)
        assert (levels_to_image(levels) == imgs).all(), "la reconstrucción no es exacta"
        counts = node_counts(levels)
        total = counts.sum(1)
        pixels = imgs.shape[1] * imgs.shape[2]
        leaves = sum(((st != SPLIT) & (st != ABSENT)).reshape(len(imgs), -1).sum(1) for st in levels)
        print(f"== {'train' if split else 'test'}: {len(imgs)} imágenes, reconstrucción exacta")
        print(f"nodos por nivel (media): {np.round(counts.mean(0), 1).tolist()}")
        print(f"nodos totales: media {total.mean():.1f}  mediana {np.median(total):.0f}  "
              f"p5 {np.percentile(total, 5):.0f}  p95 {np.percentile(total, 95):.0f}  max {total.max()}")
        print(f"hojas: media {leaves.mean():.1f}")
        print(f"nodos/píxeles: {total.mean() / pixels:.3f}   "
              f"coste de atención relativo (L^2): {(total.astype(float) ** 2).mean() / pixels ** 2:.3f}")
        # bits de información: un nodo interno 3 clases, un píxel 2 clases
        bits = (counts[:, :-1].sum(1) * np.log2(3) + counts[:, -1]).mean()
        print(f"cota de bits (codificación uniforme): árbol {bits:.0f} frente a {pixels} de la rejilla")


if __name__ == "__main__":
    main()
