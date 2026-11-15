"""Barrido de muestreo nivel a nivel de un modelo 'tile' con zonas calientes, orden y temperatura por nivel.

  python -m experiments.levels_sweep --ckpt results/qd_tile_ft/model.pt --heat -1 0 1 2 --order random conf
  python -m experiments.levels_sweep --ckpt ... --ltemps 1,1,1,0.9,0.9,0.9 0.9,0.9,0.9,0.9,0.9,0.9

Escribe results/analysis/levels_<nombre><tag>.json
"""
import argparse
import itertools
import json
import os

import numpy as np
import torch

from data.datasets import ckpt_dataset, load_images
from experiments.common import load_model, pick_device
from experiments.evaluate import embed, frechet, get_classifier
from models.tile_joint import sample_levels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--temps", type=float, nargs="+", default=[0.9])
    p.add_argument("--ltemps", nargs="+", default=None, help="temperaturas por nivel separadas por comas (sustituye a --temps)")
    p.add_argument("--steps", nargs="+", default=["8"], help="pasos por nivel; un número o una lista separada por comas")
    p.add_argument("--heat", type=int, nargs="+", default=[-1])
    p.add_argument("--radius", type=int, nargs="+", default=[1])
    p.add_argument("--order", nargs="+", default=["random"])
    p.add_argument("--lazy", type=int, nargs="+", default=[0])
    p.add_argument("--no_zero", type=int, nargs="+", default=[0])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--num", type=int, default=2000)
    p.add_argument("--batch", type=int, default=500)
    p.add_argument("--tag", default="")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device = pick_device(args.device)
    dataset = ckpt_dataset(args.ckpt)
    clf = get_classifier(device, dataset)
    f_test, _ = embed(clf, load_images(dataset, False)[0], device)
    labels = torch.arange(args.num) % 10
    model, _ = load_model(args.ckpt, device)
    temps = [[float(v) for v in t.split(",")] for t in args.ltemps] if args.ltemps else args.temps
    out = []
    for steps, heat, radius, order, nz, temp, lazy in itertools.product(args.steps, args.heat, args.radius, args.order,
                                                                        args.no_zero, temps, args.lazy):
        spl = [int(v) for v in steps.split(",")]
        spl = spl[0] if len(spl) == 1 else spl
        runs = []
        for seed in range(args.seeds):
            g = torch.Generator(device).manual_seed(seed)
            imgs, sts = [], []
            for i in range(0, args.num, args.batch):
                im, st = sample_levels(model, len(labels[i:i + args.batch]), spl, labels[i:i + args.batch], temp, device, g,
                                       heat=None if heat < 0 else heat, radius=radius, order=order, no_zero=bool(nz), lazy=lazy)
                imgs.append(im.cpu().numpy()); sts.append(st)
            feats, preds = embed(clf, np.concatenate(imgs), device)
            runs.append({"acc": 100 * float((preds == labels.numpy()).mean()), "frechet": frechet(feats, f_test),
                         **{k: float(np.mean([s[k] for s in sts])) for k in sts[0]}})
        r = {"steps": steps, "heat": heat, "radius": radius, "order": order, "no_zero": nz, "lazy": lazy, "temperatura": temp,
             "acc": round(float(np.mean([x["acc"] for x in runs])), 2),
             "frechet_min": round(min(x["frechet"] for x in runs), 2),
             "frechet_max": round(max(x["frechet"] for x in runs), 2),
             "frechet_media": round(float(np.mean([x["frechet"] for x in runs])), 2),
             **{k: round(float(np.mean([x[k] for x in runs])), 1) for k in
                ("evals", "tokens_por_muestra", "tokens_sin_cache", "pares_por_muestra", "nodos_final")}}
        out.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    os.makedirs("results/analysis", exist_ok=True)
    name = os.path.basename(os.path.dirname(args.ckpt)) + args.tag
    with open(f"results/analysis/levels_{name}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
