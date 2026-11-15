"""Barrido de muestreo de 'tile_joint': pasos, reenmascarado y zonas calientes (no entrena nada).

  python -m experiments.joint_sweep --ckpt results/qd_joint/model.pt --steps 16 32 --remask none oldest --heat -1 1 2 4

--heat -1 recalcula todas las celdas en cada paso; k >= 0 sólo las tapadas y las que tuvieron un cambio
cerca en los últimos k pasos. Escribe results/analysis/joint_<nombre><tag>.json
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
from models.tile_joint import sample


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--temps", type=float, nargs="+", default=[0.9])
    p.add_argument("--steps", type=int, nargs="+", default=[32])
    p.add_argument("--remask", nargs="+", default=["oldest"])
    p.add_argument("--heat", type=int, nargs="+", default=[-1])
    p.add_argument("--radius", type=int, nargs="+", default=[1])
    p.add_argument("--back", type=int, nargs="+", default=[1])
    p.add_argument("--rho", type=float, nargs="+", default=[2.0])
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
    out = []
    for steps, remask, heat, radius, rho, temp, back in itertools.product(args.steps, args.remask, args.heat, args.radius,
                                                                          args.rho, args.temps, args.back):
        runs = []
        for seed in range(args.seeds):
            g = torch.Generator(device).manual_seed(seed)
            imgs, sts = [], []
            for i in range(0, args.num, args.batch):
                im, st = sample(model, len(labels[i:i + args.batch]), steps, labels[i:i + args.batch], temp, device, g,
                                remask=remask, rho=rho, heat=None if heat < 0 else heat, radius=radius, back=back)
                imgs.append(im.cpu().numpy()); sts.append(st)
            feats, preds = embed(clf, np.concatenate(imgs), device)
            runs.append({"acc": 100 * float((preds == labels.numpy()).mean()), "frechet": frechet(feats, f_test),
                         **{k: float(np.mean([s[k] for s in sts])) for k in sts[0]}})
        r = {"steps": steps, "remask": remask, "heat": heat, "radius": radius, "rho": rho, "back": back, "temperatura": temp,
             "acc": round(float(np.mean([x["acc"] for x in runs])), 2),
             "frechet_min": round(min(x["frechet"] for x in runs), 2),
             "frechet_max": round(max(x["frechet"] for x in runs), 2),
             "frechet_media": round(float(np.mean([x["frechet"] for x in runs])), 2),
             **{k: round(float(np.mean([x[k] for x in runs])), 1) for k in
                ("evals", "tokens_por_muestra", "tokens_sin_cache", "pares_por_muestra", "reenmascarados", "nodos_final")}}
        out.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    os.makedirs("results/analysis", exist_ok=True)
    name = os.path.basename(os.path.dirname(args.ckpt)) + args.tag
    with open(f"results/analysis/joint_{name}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
