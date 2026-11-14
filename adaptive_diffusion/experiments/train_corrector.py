"""Entrena un corrector aparte para un 'grow' ya entrenado, sin tocar el generador.

El generador decide un nivel con parte de los nodos tapados; el corrector ve ese nivel ya
decidido entero (sin máscaras) y predice el estado verdadero de cada nodo del nivel. Al
muestrear, un nodo al que el corrector da poca probabilidad se cambia por lo que dice el
corrector, antes de crear los hijos: una subdivisión mala se une antes de pagar sus 4 hijos.

Los errores de entrenamiento son los del propio generador: se tapa parte del nivel, se
sortea su predicción (temperatura al azar) y se sustituye.

  python -m experiments.train_corrector --gen results/qd_grow_ft_ctrl/model.pt --out results/qd_corr
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.quadtree_dataset import QuadtreeMNIST
from experiments.common import load_model, pick_device
from models.adaptive_diffusion import make_training_batch
from models.quadtree_transformer import PAD, QuadtreeTransformer


@torch.no_grad()
def corrupt(gen, b, labels, generator=None):
    """Rellena las máscaras del lote con lo que sortea el generador. Devuelve (estado, verdad, nivel en curso)."""
    masked = b["target"] != -100
    temp = 0.6 + 0.4 * torch.rand(len(labels), 1, 1, device=labels.device, generator=generator)
    logits = gen(b["state"], b["level"], b["y"], b["x"], b["pad_mask"], labels).float()
    drawn = torch.multinomial(F.softmax(logits / temp, -1).flatten(0, 1), 1, generator=generator).view(masked.shape)
    truth = torch.where(masked, b["target"], b["state"])
    cur = (b["level"] == b["level"].masked_fill(b["pad_mask"], -1).max(1, keepdim=True).values) & ~b["pad_mask"]
    return torch.where(masked, drawn, b["state"]), truth, cur


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen", required=True, help="checkpoint del generador 'grow' (no se modifica)")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    gen, meta = load_model(args.gen, device)
    assert meta["method"] == "grow"
    a, depth = meta["args"], meta["depth"]
    model = QuadtreeTransformer(depth=depth, dim=a["dim"], layers=a["layers"], heads=a["heads"]).to(device)
    model.load_state_dict(gen.state_dict())          # parte de los pesos del generador
    data = QuadtreeMNIST(train=True, dataset=meta["dataset"])
    loader = DataLoader(data, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=args.workers,
                        persistent_workers=True,
                        collate_fn=lambda b: (make_training_batch([t for t, _ in b], depth), torch.stack([y for _, y in b])))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    step, t0, agg = 0, time.time(), torch.zeros(5)
    while step < args.steps:
        for b, labels in loader:
            b = {k: v.to(device) for k, v in b.items()}
            labels = labels.to(device)
            labels = torch.where(torch.rand(len(labels), device=device) < 0.1, torch.full_like(labels, model.null_class), labels)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                state, truth, cur = corrupt(gen, b, labels)
                logits = model(state, b["level"], b["y"], b["x"], b["pad_mask"], labels).float()
            loss = F.cross_entropy(logits[cur], truth[cur])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            with torch.no_grad():
                wrong, right = cur & (state != truth), cur & (state == truth)
                hot = F.softmax(logits, -1).gather(-1, state.clamp(max=2)[..., None])[..., 0] < 0.5
                agg += torch.tensor([loss.item(), (wrong & hot).sum().item(), wrong.sum().item(),
                                     (right & hot).sum().item(), right.sum().item()])
            step += 1
            if step % 100 == 0:
                print(f"paso {step:6d} | pérdida {agg[0] / 100:.4f} | detección {100 * agg[1] / max(agg[2], 1):.1f}% | "
                      f"falsa alarma {100 * agg[3] / max(agg[4], 1):.2f}% | {time.time() - t0:.0f}s", flush=True)
                agg.zero_()
            if step >= args.steps:
                break
    torch.save({"model": model.state_dict(), "args": {**a, "method": "corrector", "gen": args.gen, **vars(args)},
                "depth": depth, "prior": None}, os.path.join(args.out, "model.pt"))
    print("guardado", os.path.join(args.out, "model.pt"))


if __name__ == "__main__":
    main()
