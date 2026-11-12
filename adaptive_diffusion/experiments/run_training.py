"""
Entrena la difusión enmascarada sobre quadtrees (variante A) en MNIST binario.
Uso (desde adaptive_diffusion/):
    python -m experiments.run_training --steps 20000 --out runs/qdd
"""
import argparse
import os
import time
import torch
from torch.utils.data import DataLoader

from data.quadtree_dataset import QuadtreeMNIST
from models.adaptive_diffusion import loss_fn, make_training_batch
from models.quadtree_transformer import QuadtreeTransformer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="usar sólo las primeras N imágenes")
    p.add_argument("--cond_drop", type=float, default=0.1, help="prob. de entrenar sin etiqueta")
    p.add_argument("--out", default="runs/qdd")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = QuadtreeMNIST(train=True, limit=args.limit)
    loader = DataLoader(data, batch_size=args.batch, shuffle=True, drop_last=True,
                        collate_fn=lambda b: ([t for t, _ in b], torch.stack([y for _, y in b])))
    model = QuadtreeTransformer(depth=data.depth, dim=args.dim, layers=args.layers, heads=args.heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    print(f"dispositivo {device} | {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parámetros | {len(data)} árboles")

    os.makedirs(args.out, exist_ok=True)
    step, t0, running = 0, time.time(), 0.0
    while step < args.steps:
        for tokens, labels in loader:
            batch = {k: v.to(device) for k, v in make_training_batch(tokens, data.depth).items()}
            labels = labels.to(device)
            drop = torch.rand(len(labels), device=device) < args.cond_drop
            labels = torch.where(drop, torch.full_like(labels, model.null_class), labels)
            loss = loss_fn(model, batch, labels)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            running += loss.item()
            if step % 100 == 0:
                print(f"paso {step:6d} | pérdida {running / 100:.4f} | {time.time() - t0:.0f}s", flush=True)
                running = 0.0
            if step >= args.steps:
                break
    torch.save({"model": model.state_dict(), "args": vars(args), "depth": data.depth},
               os.path.join(args.out, "model.pt"))
    print("guardado", os.path.join(args.out, "model.pt"))


if __name__ == "__main__":
    main()
