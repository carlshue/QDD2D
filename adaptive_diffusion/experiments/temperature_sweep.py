"""Barrido de temperatura de muestreo sobre un checkpoint ya entrenado (no entrena nada).

  python -m experiments.temperature_sweep --ckpt results/gpu01/model.pt --steps 8 10
  python -m experiments.temperature_sweep --ckpt results/edit_cell/model.pt --steps 16 32 --resolve

Escribe results/analysis/temperature_<nombre>[_resolve].json
"""
import argparse
import json
import os

import numpy as np
import torch

from data.datasets import ckpt_dataset, load_images
from experiments.common import generate, load_model, pick_device
from experiments.evaluate import embed, frechet, get_classifier


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--temps", type=float, nargs="+", default=[1.0, 0.95, 0.9, 0.85, 0.8, 0.7])
    p.add_argument("--steps", type=int, nargs="+", default=[None])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--num", type=int, default=2000)
    p.add_argument("--resolve", action="store_true")
    p.add_argument("--revise", type=float, default=1.0)
    p.add_argument("--revise_thr", type=float, default=0.5)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = pick_device(args.device)
    dataset = ckpt_dataset(args.ckpt)
    clf = get_classifier(device, dataset)
    test, _ = load_images(dataset, False)
    f_test, _ = embed(clf, test, device)
    labels = torch.arange(args.num) % 10
    model, meta = load_model(args.ckpt, device)
    out = []
    for steps in args.steps:
        for temp in args.temps:
            runs = []
            for seed in range(args.seeds):
                torch.manual_seed(seed)
                imgs, st = generate(model, meta, labels, device, steps, temp, resolve_fresh=args.resolve,
                                    revise=args.revise, revise_thr=args.revise_thr)
                feats, preds = embed(clf, imgs, device)
                runs.append({"acc": 100 * float((preds == labels.numpy()).mean()), "frechet": frechet(feats, f_test), **st})
            r = {"metodo": meta["method"], "resolve": args.resolve, "steps": steps, "temperatura": temp,
                 "acc_min": round(min(x["acc"] for x in runs), 2), "acc_max": round(max(x["acc"] for x in runs), 2),
                 "frechet_min": round(min(x["frechet"] for x in runs), 2),
                 "frechet_max": round(max(x["frechet"] for x in runs), 2),
                 "frechet_media": round(float(np.mean([x["frechet"] for x in runs])), 2),
                 **{k: round(float(np.mean([x[k] for x in runs])), 2)
                    for k in ("revisiones", "uniones", "subdivisiones_tardias") if k in runs[0]},
                 "evals": round(float(np.mean([x["evals"] for x in runs])), 1),
                 "tokens_por_muestra": round(float(np.mean([x["tokens_por_muestra"] for x in runs]))),
                 "nodos_final": round(float(np.mean([x["nodos_final"] for x in runs])), 1)}
            out.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
    os.makedirs("results/analysis", exist_ok=True)
    name = os.path.basename(os.path.dirname(args.ckpt)) + ("_resolve" if args.resolve else "") \
        + ("_sin_union" if args.revise <= 0 else "") + (f"_thr{args.revise_thr:g}" if args.revise_thr != 0.5 else "")
    with open(f"results/analysis/temperature_{name}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
