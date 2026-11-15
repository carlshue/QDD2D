"""Entrena 'tile_joint': tokens de ocupación con todos los niveles tapados a la vez (models/tile_joint.py).

  python -m experiments.train_joint --out results/qd_joint --seed 0

Misma receta que 'tile': 30 000 pasos a lr 1e-3 y 5 000 a 3e-4, lote 256. Los lotes se preparan en la gráfica.
"""
import argparse
import os
import time

import torch

from data.datasets import load_images
from experiments.common import pick_device
from models.quadtree_transformer import QuadtreeTransformer
from models.tile_joint import Grid, loss_fn, make_training_batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="quickdraw")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--steps_low", type=int, default=5000, help="pasos finales a --lr_low")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_low", type=float, default=3e-4)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--cond_drop", type=float, default=0.1)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    args.method = "tile_joint"
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(args.out, exist_ok=True)
    imgs, labels = load_images(args.dataset, True)
    imgs, labels = torch.from_numpy(imgs).to(device), torch.from_numpy(labels).to(device)
    depth = int(imgs.shape[1]).bit_length() - 1
    grid = Grid(depth, device)
    model = QuadtreeTransformer(depth=depth, dim=args.dim, layers=args.layers, heads=args.heads, num_states=16).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    total = args.steps + args.steps_low

    def save():
        torch.save({"model": model.state_dict(), "args": vars(args), "depth": depth, "prior": None},
                   os.path.join(args.out, "model.pt"))
    print(f"tile_joint | {sum(q.numel() for q in model.parameters()) / 1e6:.2f} M parámetros | {len(imgs)} dibujos", flush=True)
    t0, run, toks = time.time(), 0.0, 0
    for step in range(1, total + 1):
        if step == args.steps + 1:
            for g in opt.param_groups:
                g["lr"] = args.lr_low
        idx = torch.randint(0, len(imgs), (args.batch,), device=device)
        batch = make_training_batch(grid, imgs[idx])
        lab = labels[idx]
        lab = torch.where(torch.rand(len(lab), device=device) < args.cond_drop, torch.full_like(lab, model.null_class), lab)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            loss = loss_fn(model, batch, lab)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        run += loss.item(); toks += int((~batch["pad_mask"]).sum())
        if step % 500 == 0:
            el = time.time() - t0
            print(f"paso {step:6d} | pérdida {run / 500:.4f} | {toks / 500 / args.batch:.0f} tokens/árbol | {el:.0f}s | "
                  f"faltan ~{el / step * (total - step) / 60:.0f} min", flush=True)
            run, toks = 0.0, 0
        if step % 5000 == 0:
            save()
    save()
    print("guardado", os.path.join(args.out, "model.pt"))


if __name__ == "__main__":
    main()
