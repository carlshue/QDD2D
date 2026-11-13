"""
Genera imágenes con un modelo entrenado (de cualquiera de los dos métodos) y
guarda una rejilla PNG, una columna por dígito.
Uso (desde adaptive_diffusion/):
    python -m experiments.run_sampling --ckpt runs/qdd/model.pt --num 100
"""
import argparse
import os
import numpy as np
import torch

from experiments.common import generate, load_model, pick_device


def save_grid(imgs, path, cols=10):
    rows = int(np.ceil(len(imgs) / cols))
    s = imgs.shape[1]
    grid = np.zeros((rows * (s + 2), cols * (s + 2)), dtype=np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        grid[r * (s + 2) + 1: r * (s + 2) + 1 + s, c * (s + 2) + 1: c * (s + 2) + 1 + s] = im * 255
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.imsave(path, grid, cmap="gray")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--num", type=int, default=100)
    p.add_argument("--steps", type=int, default=None,
                   help="pasos de muestreo: por nivel en 'grow' y 'grow_merge' (def. 8), totales en 'edit' (def. 32)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--resolve", action="store_true",
                   help="sólo 'edit': resolver con la red los nodos recién creados en cada paso (mejor calidad, ~3,5x evaluaciones)")
    p.add_argument("--revise", type=float, default=1.0,
                   help="sólo 'grow_merge': prob. de revisar un nodo dudoso ya decidido (0 = sin unión)")
    p.add_argument("--revise_thr", type=float, default=0.5,
                   help="sólo 'grow_merge': un nodo es dudoso si la red da a su estado menos de esta prob.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="auto", help="auto, cuda, mps o cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    model, meta = load_model(args.ckpt, device)
    imgs, stats = generate(model, meta, torch.arange(args.num) % 10, device, args.steps, args.temperature,
                           resolve_fresh=args.resolve,
                           revise=args.revise, revise_thr=args.revise_thr)
    print(f"método {meta['method']} | {args.num} muestras | " +
          " | ".join(f"{k} {v:.1f}" for k, v in stats.items()), flush=True)
    out = args.out or os.path.join(os.path.dirname(args.ckpt), "samples.png")
    save_grid(imgs, out)
    print("guardado", out)


if __name__ == "__main__":
    main()
