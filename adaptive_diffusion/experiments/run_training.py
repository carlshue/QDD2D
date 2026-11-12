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


def pick_device(name="auto"):
    if name != "auto":
        if name.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit("Se pidió CUDA pero PyTorch no detecta ninguna GPU.")
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save(model, args, depth):
    torch.save({"model": model.state_dict(), "args": vars(args), "depth": depth},
               os.path.join(args.out, "model.pt"))


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
    p.add_argument("--device", default="auto", help="auto, cuda, cuda:0, mps o cpu")
    p.add_argument("--no_amp", action="store_true", help="desactivar precisión mixta en GPU")
    p.add_argument("--save_every", type=int, default=2000)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print("construyendo los árboles (unos 20-60 s)...", flush=True)
    use_amp = device.startswith("cuda") and not args.no_amp
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    data = QuadtreeMNIST(train=True, limit=args.limit)
    loader = DataLoader(data, batch_size=args.batch, shuffle=True, drop_last=True,
                        pin_memory=device.startswith("cuda"),
                        collate_fn=lambda b: ([t for t, _ in b], torch.stack([y for _, y in b])))
    model = QuadtreeTransformer(depth=data.depth, dim=args.dim, layers=args.layers, heads=args.heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    name = torch.cuda.get_device_name(device) if device.startswith("cuda") else device
    print(f"dispositivo {name} | precisión mixta {'sí' if use_amp else 'no'} | {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parámetros | {len(data)} árboles", flush=True)

    os.makedirs(args.out, exist_ok=True)
    step, t0, running = 0, time.time(), 0.0
    while step < args.steps:
        for tokens, labels in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in make_training_batch(tokens, data.depth).items()}
            labels = labels.to(device, non_blocking=True)
            drop = torch.rand(len(labels), device=device) < args.cond_drop
            labels = torch.where(drop, torch.full_like(labels, model.null_class), labels)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = loss_fn(model, batch, labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            running += loss.item()
            if step in (1, 10):
                print(f"paso {step:6d} | pérdida {loss.item():.4f} | {time.time() - t0:.0f}s", flush=True)
            if step % 100 == 0:
                eta = (time.time() - t0) / step * (args.steps - step) / 60
                print(f"paso {step:6d} | pérdida {running / 100:.4f} | {time.time() - t0:.0f}s | faltan ~{eta:.0f} min", flush=True)
                running = 0.0
            if step % args.save_every == 0:
                save(model, args, data.depth)
            if step >= args.steps:
                break
    save(model, args, data.depth)
    print("guardado", os.path.join(args.out, "model.pt"))


if __name__ == "__main__":
    main()
