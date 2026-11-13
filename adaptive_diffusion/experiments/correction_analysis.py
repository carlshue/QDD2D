"""Medida directa de la revisión en 'grow_merge': ¿detecta y arregla la red los errores?

Sobre imágenes de prueba se construyen árboles parciales con errores en un nivel ya
decidido y se mira qué dice la red de esos nodos, sin generar nada:

  deteccion     % de nodos erróneos a cuyo estado la red da menos de `thr` de probabilidad
  falsa_alarma  % de nodos correctos a los que les pasa lo mismo
  arreglo       % de nodos erróneos para los que el estado más probable es el verdadero

Los errores los comete una red "fuente" (--source; por defecto, cada modelo los suyos):
se tapa parte de un nivel y se sortea su predicción, como al muestrear.

  python -m experiments.correction_analysis --ckpt results/grow_merge/model.pt results/grow_merge_self/model.pt
  python -m experiments.correction_analysis --ckpt A.pt B.pt --source A.pt     # mismos errores para todos
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F

from data.datasets import ckpt_dataset, load_images
from experiments.common import load_model, pick_device
from models.edit_diffusion import _gather
from models.grow_merge import own_errors
from models.quadtree_transformer import MASK
from structures.binary_quadtree import SPLIT
from structures.pyramid import Pyramid


@torch.no_grad()
def measure(model, source, pyr, s0, labels, thr=0.5, generator=None):
    n, dev = s0.shape[0], s0.device
    lev = pyr.level[None]
    keys = ("erroneos", "detectados", "arreglados", "correctos", "falsas_alarmas")
    kinds = ("split_de_mas", "split_de_menos", "relleno")
    tot = {k: 0 for k in keys}
    per = {k: {"erroneos": 0, "detectados": 0, "arreglados": 0} for k in kinds}
    for l in range(1, pyr.depth + 1):
        lvl = torch.full((n, 1), l, device=dev)
        state = own_errors(source, pyr, s0, lvl, labels, generator)
        t = torch.full((n,), 0.5, device=dev)
        hide = (lev == lvl) & (torch.rand(s0.shape, device=dev, generator=generator) < 0.5)
        state = torch.where(hide, torch.full_like(state, MASK), state)
        ex = pyr.exists(state) & (lev <= lvl)
        order, pad = pyr.to_sequence(ex)
        b = _gather(pyr, state, order, pad)
        probs = F.softmax(model(b["state"], b["level"], b["y"], b["x"], pad, labels, t).float(), -1)
        truth = s0.gather(1, order).long()
        decided = ~pad & (b["state"] != MASK) & (b["level"] < l)
        cur = b["state"].clamp(max=probs.shape[-1] - 1)
        hot = probs.gather(-1, cur[..., None])[..., 0] < thr
        wrong = decided & (b["state"] != truth)
        right = decided & (b["state"] == truth)
        fixed = wrong & (probs.argmax(-1) == truth)
        tot["erroneos"] += int(wrong.sum()); tot["detectados"] += int((wrong & hot).sum())
        tot["arreglados"] += int(fixed.sum())
        tot["correctos"] += int(right.sum()); tot["falsas_alarmas"] += int((right & hot).sum())
        masks = {"split_de_mas": wrong & (b["state"] == SPLIT),
                 "split_de_menos": wrong & (truth == SPLIT),
                 "relleno": wrong & (b["state"] != SPLIT) & (truth != SPLIT)}
        for k, m in masks.items():
            per[k]["erroneos"] += int(m.sum()); per[k]["detectados"] += int((m & hot).sum())
            per[k]["arreglados"] += int((m & fixed).sum())

    def pct(a, b_):
        return round(100 * a / max(b_, 1), 1)
    res = {"umbral": thr, "errores_por_arbol": round(tot["erroneos"] / (n * pyr.depth), 2),
           "tasa_error_pct": pct(tot["erroneos"], tot["erroneos"] + tot["correctos"]),
           "deteccion_pct": pct(tot["detectados"], tot["erroneos"]),
           "falsa_alarma_pct": pct(tot["falsas_alarmas"], tot["correctos"]),
           "arreglo_pct": pct(tot["arreglados"], tot["erroneos"])}
    for k, v in per.items():
        res[k] = {"n": v["erroneos"], "deteccion_pct": pct(v["detectados"], v["erroneos"]),
                  "arreglo_pct": pct(v["arreglados"], v["erroneos"])}
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--source", default=None, help="checkpoint que comete los errores (por defecto, cada modelo los suyos)")
    p.add_argument("--num", type=int, default=5000)
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device = pick_device(args.device)
    dataset = ckpt_dataset(args.ckpt[0])
    imgs, labels = load_images(dataset, False)
    imgs, labels = imgs[:args.num], labels[:args.num]
    src = load_model(args.source, device)[0] if args.source else None
    out = {}
    for path in args.ckpt:
        model, meta = load_model(path, device)
        if meta["method"] != "grow_merge":
            print(f"{path}: sólo tiene sentido para 'grow_merge', se omite")
            continue
        pyr = Pyramid(meta["depth"], device)
        s0 = pyr.from_images(torch.from_numpy(imgs).to(device))
        g = torch.Generator(device).manual_seed(args.seed)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
            res = measure(model, src or model, pyr, s0, torch.from_numpy(labels).to(device), args.thr, g)
        res["fuente_de_errores"] = args.source or "propio"
        out[path] = res
        print(path, json.dumps(res, ensure_ascii=False), flush=True)
    os.makedirs("results/analysis", exist_ok=True)
    tag = ("comun" if args.source else "propio") + ("" if dataset == "mnist" else f"_{dataset}")
    with open(f"results/analysis/correction_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
