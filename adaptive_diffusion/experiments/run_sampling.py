"""
Genera imágenes con un modelo entrenado y guarda una rejilla PNG.
Uso (desde adaptive_diffusion/):
    python -m experiments.run_sampling --ckpt runs/qdd/model.pt --num 64
"""
import argparse
import os
import numpy as np
import torch

from models.adaptive_diffusion import sample
from models.quadtree_transformer import QuadtreeTransformer
from structures.binary_quadtree import levels_to_image, tokens_to_levels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--num", type=int, default=64)
    p.add_argument("--steps_per_level", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device)
    a = ck["args"]
    model = QuadtreeTransformer(depth=ck["depth"], dim=a["dim"], layers=a["layers"], heads=a["heads"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    labels = torch.arange(args.num) % 10   # una columna por dígito
    trees, evals = sample(model, args.num, args.steps_per_level, labels, args.temperature, device)
    imgs = np.stack([levels_to_image([lv[None] for lv in tokens_to_levels(t.numpy(), ck["depth"])])[0] for t in trees])
    sizes = np.array([len(t) for t in trees])
    print(f"{args.num} muestras | nodos por árbol: media {sizes.mean():.1f}, max {sizes.max()} "
          f"(rejilla: {imgs.shape[1] * imgs.shape[2]}) | evaluaciones de red: {evals}")

    cols = 10
    rows = int(np.ceil(args.num / cols))
    s = imgs.shape[1]
    grid = np.zeros((rows * (s + 2), cols * (s + 2)), dtype=np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        grid[r * (s + 2) + 1: r * (s + 2) + 1 + s, c * (s + 2) + 1: c * (s + 2) + 1 + s] = im * 255
    out = args.out or os.path.join(os.path.dirname(args.ckpt), "samples.png")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.imsave(out, grid, cmap="gray")
    print("guardado", out)


if __name__ == "__main__":
    main()
