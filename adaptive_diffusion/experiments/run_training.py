"""
Entrena un modelo sobre quadtrees de MNIST binario.

  --method grow   el árbol crece de la raíz a los píxeles, tapando y destapando
                  nodos nivel a nivel (el prototipo original).
  --method grow_merge  crece como 'grow', pero los nodos ya decididos se pueden
                  revisar: subdividir más tarde o juntar. Sin árbol aleatorio.
  --method edit   se parte de un árbol aleatorio y se edita: subdividir, juntar
                  o cambiar el relleno de cualquier nodo en cualquier paso.
                  --prior elige el tipo de árbol aleatorio (uniform, level, cell).

Uso (desde adaptive_diffusion/):
    python -m experiments.run_training --method edit --prior cell --out runs/edit
"""
import argparse
import os
import time
import torch

from experiments.common import pick_device
from models.quadtree_transformer import QuadtreeTransformer


def save(model, args, depth, prior=None):
    torch.save({"model": model.state_dict(), "args": vars(args), "depth": depth,
                "prior": None if prior is None else prior.cpu()}, os.path.join(args.out, "model.pt"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["grow", "grow_merge", "edit"], default="grow")
    p.add_argument("--prior", choices=["uniform", "level", "cell"], default="cell",
                   help="árbol aleatorio de partida (sólo --method edit)")
    p.add_argument("--prior_p", type=float, default=0.667, help="prob. de subdividir del ruido 'uniform'")
    p.add_argument("--eps_max", type=float, default=0.15,
                   help="sólo grow_merge: fracción máxima de nodos ya decididos que se estropean al entrenar")
    p.add_argument("--corrupt", choices=["prior", "self", "both"], default="prior",
                   help="sólo grow_merge: los nodos decididos se estropean con el ruido (prior), con los errores "
                        "de la propia red (self) o con ambos (both)")
    p.add_argument("--dataset", choices=["mnist", "quickdraw"], default="mnist")
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
    p.add_argument("--workers", type=int, default=0,
                   help="sólo grow: procesos que preparan los lotes (0 = en el proceso principal, como antes)")
    p.add_argument("--save_every", type=int, default=2000)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    use_amp = device.startswith("cuda") and not args.no_amp
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out, exist_ok=True)
    print("preparando los datos...", flush=True)

    prior = None
    if args.method in ("edit", "grow_merge"):
        from data.datasets import load_images
        from structures.pyramid import Pyramid
        if args.method == "edit":
            from models.edit_diffusion import loss_fn, training_batch
        else:
            from models.grow_merge import loss_fn, training_batch as _tb

            def training_batch(pyr, s0, prior, label=None):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    return _tb(pyr, s0, prior, args.eps_max, model=model, label=label, corrupt=args.corrupt)
        imgs, labels_np = load_images(args.dataset, True)
        if args.limit:
            imgs, labels_np = imgs[:args.limit], labels_np[:args.limit]
        depth = int(imgs.shape[1]).bit_length() - 1
        pyr = Pyramid(depth, device)
        s0 = pyr.from_images(torch.from_numpy(imgs).to(device))
        all_labels = torch.from_numpy(labels_np).to(device)
        if args.method == "grow_merge":   # el ruido sólo decide con qué se estropean los nodos ya decididos
            prior = pyr.prior_from_data(s0, "cell")
            print(f"crecimiento con unión: nodos decididos estropeados con '{args.corrupt}' "
                  f"(ruido hasta {100 * args.eps_max:.0f}%)", flush=True)
        else:
            prior = pyr.prior_uniform(args.prior_p) if args.prior == "uniform" else pyr.prior_from_data(s0, args.prior)
            noise_nodes = pyr.exists(pyr.sample_states(prior, 2000)).sum(1).float().mean().item()
            print(f"ruido '{args.prior}': {noise_nodes:.0f} nodos de media por árbol aleatorio", flush=True)
        n_data = len(s0)

        def batches():
            while True:
                idx = torch.randint(0, n_data, (args.batch,), device=device)
                if args.method == "grow_merge":
                    yield training_batch(pyr, s0[idx], prior, all_labels[idx]), all_labels[idx]
                else:
                    yield training_batch(pyr, s0[idx], prior), all_labels[idx]
    else:
        from torch.utils.data import DataLoader
        from data.quadtree_dataset import QuadtreeMNIST
        from models.adaptive_diffusion import loss_fn as grow_loss, make_training_batch
        data = QuadtreeMNIST(train=True, limit=args.limit, dataset=args.dataset)
        depth, n_data = data.depth, len(data)
        if args.workers > 0:     # construir el lote (tapar nodos, rellenar) en procesos aparte
            def collate(b):
                return make_training_batch([t for t, _ in b], depth), torch.stack([y for _, y in b])
        else:
            def collate(b):
                return [t for t, _ in b], torch.stack([y for _, y in b])
        loader = DataLoader(data, batch_size=args.batch, shuffle=True, drop_last=True, collate_fn=collate,
                            num_workers=args.workers, persistent_workers=args.workers > 0)

        def loss_fn(model, batch, labels):
            return grow_loss(model, batch, labels), None

        def batches():
            while True:
                for tokens, labels in loader:
                    b = tokens if args.workers > 0 else make_training_batch(tokens, depth)
                    yield {k: v.to(device) for k, v in b.items()}, labels.to(device)

    model = QuadtreeTransformer(depth=depth, dim=args.dim, layers=args.layers, heads=args.heads,
                                use_time=(args.method != "grow")).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    name = torch.cuda.get_device_name(device) if device.startswith("cuda") else device
    print(f"método {args.method} | dispositivo {name} | precisión mixta {'sí' if use_amp else 'no'} | "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parámetros | {n_data} árboles", flush=True)

    t0, running, running_acc = time.time(), 0.0, 0.0
    for step, (batch, labels) in enumerate(batches(), start=1):
        drop = torch.rand(len(labels), device=device) < args.cond_drop
        labels = torch.where(drop, torch.full_like(labels, model.null_class), labels)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss, acc = loss_fn(model, batch, labels)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        running += loss.item()
        running_acc += 0.0 if acc is None else acc.item()
        if step in (1, 10):
            print(f"paso {step:6d} | pérdida {loss.item():.4f} | {time.time() - t0:.0f}s", flush=True)
        if step % 100 == 0:
            eta = (time.time() - t0) / step * (args.steps - step) / 60
            extra = "" if acc is None else f" | nodos ruidosos corregidos {100 * running_acc / 100:.1f}%"
            print(f"paso {step:6d} | pérdida {running / 100:.4f}{extra} | {time.time() - t0:.0f}s | "
                  f"faltan ~{eta:.0f} min", flush=True)
            running, running_acc = 0.0, 0.0
        if step % args.save_every == 0:
            save(model, args, depth, prior)
        if step >= args.steps:
            break
    save(model, args, depth, prior)
    print("guardado", os.path.join(args.out, "model.pt"))


if __name__ == "__main__":
    main()
