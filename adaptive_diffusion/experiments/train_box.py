"""
Entrena la generación nivel a nivel sobre árboles de cajas 3D (ABO voxelizado).

    --tree tile   octree de la bibliografía: un byte de ocupación por nodo
    --tree oct    octree con hojas llenas a cualquier nivel
    --tree omni   omnitree: cada nodo elige qué ejes partir

Mismo modelo, mismos datos y misma receta para los tres; sólo cambia el árbol.

Uso (desde adaptive_diffusion/):
    python -m experiments.train_box --tree omni --out results/box_omni
"""
import argparse
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from experiments.box_common import make_dataset, make_model
from experiments.common import pick_device
from models.box_grow import BucketSampler, make_training_batch

LN2 = math.log(2)


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    t = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (args.final_lr + (1 - args.final_lr) * 0.5 * (1 + math.cos(math.pi * t)))


@torch.no_grad()
def held_out_loss(model, data, device, use_amp, n=512, batch=64):
    """Pérdida sobre formas de prueba, con un sorteo fijo: sirve para ver el sobreajuste."""
    model.eval()
    g = torch.Generator().manual_seed(0)
    total, count = 0.0, 0
    for i in range(0, min(n, len(data)), batch):
        items = [data[j] for j in range(i, min(i + batch, n, len(data)))]
        b, labels = make_training_batch(items, g)
        b = {k: v.to(device) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss, _ = model.loss(b, labels.to(device))
        m = int(b["masked"].sum())
        total, count = total + loss.item() * m, count + m
    model.train()
    return total / max(count, 1) / LN2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tree", choices=["tile", "oct", "omni"], required=True)
    p.add_argument("--res", type=int, default=32)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--accum", type=int, default=2, help="lotes por paso (lote efectivo = batch × accum)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--final_lr", type=float, default=0.1, help="fracción de lr al final (decaimiento coseno)")
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--aug", type=int, default=8, help="variantes por forma: giros de 90° y espejo (1, 2, 4 u 8)")
    p.add_argument("--max_tile_tokens", type=int, default=2048,
                   help="descarta las formas cuyo octree `tile` supera este tamaño (igual para todos los árboles)")
    p.add_argument("--limit", type=int, default=None, help="usar sólo las primeras N formas")
    p.add_argument("--cond_drop", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--init", default=None, help="checkpoint del que partir")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    use_amp = device.startswith("cuda") and not args.no_amp
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out, exist_ok=True)

    print("preparando los árboles (la primera vez tarda; luego se leen de data/abo/cache)...", flush=True)
    data = make_dataset(args.tree, args.res, True, args.aug, args.max_tile_tokens, args.limit,
                        workers=max(args.workers, 1) * 2)
    val = make_dataset(args.tree, args.res, False, 1, args.max_tile_tokens, log=None)
    lengths = [data.length(i) for i in range(len(data))]
    print(f"árbol '{args.tree}' a {args.res}³: {len(data)} árboles ({len(data) // args.aug} formas × {args.aug}), "
          f"{sum(lengths) / len(lengths):.0f} tokens de media, máx. {max(lengths)}", flush=True)
    loader = DataLoader(data, batch_sampler=BucketSampler(lengths, args.batch, seed=args.seed),
                        collate_fn=make_training_batch, num_workers=args.workers,
                        persistent_workers=args.workers > 0)

    def batches():
        while True:
            yield from loader

    model = make_model(args.tree, data.depth, args.dim, args.layers, args.heads).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device)["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    name = torch.cuda.get_device_name(device) if device.startswith("cuda") else device
    print(f"dispositivo {name} | precisión mixta {'sí' if use_amp else 'no'} | "
          f"{sum(q.numel() for q in model.parameters()) / 1e6:.2f} M parámetros | lote {args.batch} × {args.accum}",
          flush=True)

    def save():
        torch.save({"model": model.state_dict(), "args": vars(args), "depth": data.depth},
                   os.path.join(args.out, "model.pt"))

    stream = batches()
    t0, run, parts_run, seen = time.time(), 0.0, {}, 0
    for step in range(1, args.steps + 1):
        for group in opt.param_groups:
            group["lr"] = lr_at(step - 1, args)
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            b, labels = next(stream)
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            labels = labels.to(device)
            drop = torch.rand(len(labels), device=device) < args.cond_drop
            labels = torch.where(drop, torch.full_like(labels, model.null_class), labels)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss, parts = model.loss(b, labels)
            scaler.scale(loss / args.accum).backward()
            run += loss.item() / args.accum
            seen += int((~b["pad"]).sum())
            for k, v in parts.items():
                parts_run[k] = parts_run.get(k, 0.0) + v.item() / args.accum
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step in (1, 10) or step % 100 == 0:
            n = 1 if step in (1, 10) else 100
            if step in (1, 10):
                shown, detail = loss.item() / LN2, ""
            else:
                shown = run / n / LN2
                detail = " (" + ", ".join(f"{k} {v / n / LN2:.2f}" for k, v in parts_run.items()) + ")"
            eta = (time.time() - t0) / step * (args.steps - step) / 60
            print(f"paso {step:6d} | pérdida {shown:.3f} bits/token{detail} | {time.time() - t0:.0f}s | "
                  f"faltan ~{eta:.0f} min", flush=True)
            if step % 100 == 0:
                run, parts_run = 0.0, {}
        if step % args.val_every == 0 or step == args.steps:
            print(f"paso {step:6d} | pérdida en prueba {held_out_loss(model, val, device, use_amp):.3f} bits/token | "
                  f"{seen / step / args.accum:.0f} tokens por lote", flush=True)
        if step % args.save_every == 0:
            save()
    save()
    print("guardado", os.path.join(args.out, "model.pt"))


if __name__ == "__main__":
    main()
