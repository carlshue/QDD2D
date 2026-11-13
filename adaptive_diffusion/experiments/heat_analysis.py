"""
Medición pasiva de nodos calientes: ¿cuánto cómputo se ahorraría si en cada
paso sólo se volvieran a mirar las zonas donde la red aún duda?

No cambia las muestras; sólo anota (ver models/heat.py). Vale para los métodos
que vuelven a mirar nodos: 'edit' y 'grow_merge'.

Uso (desde adaptive_diffusion/):
    python -m experiments.heat_analysis --ckpt results/edit_cell/model.pt --resolve
    python -m experiments.heat_analysis --ckpt results/grow_merge/model.pt
Escribe results/analysis/heat_<experimento><variante>.json
"""
import argparse
import json
import os
import torch

from experiments.common import generate, load_model, pick_device
from models.heat import HeatTrace
from structures.pyramid import Pyramid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--num", type=int, default=1000)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--resolve", action="store_true")
    p.add_argument("--revise", type=float, default=1.0)
    p.add_argument("--revise_thr", type=float, default=0.5)
    p.add_argument("--thr", type=float, nargs="+", default=[0.5, 0.9, 0.99],
                   help="un nodo es caliente si la red da a su estado actual menos de esta prob.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out_dir", default="results/analysis")
    args = p.parse_args()

    device = pick_device(args.device)
    labels = torch.arange(args.num) % 10
    os.makedirs(args.out_dir, exist_ok=True)
    for path in args.ckpt:
        model, meta = load_model(path, device)
        if meta["method"] == "grow":
            print(f"{path}: 'grow' decide cada nodo una sola vez; no hay nada que volver a mirar. Se salta.")
            continue
        pyr = Pyramid(meta["depth"], device)
        res = {"modelo": path, "metodo": meta["method"], "muestras": args.num, "pasos": args.steps,
               "resolve": args.resolve, "revise": args.revise, "revise_thr": args.revise_thr, "umbrales": {}}
        for thr in args.thr:
            torch.manual_seed(args.seed)       # misma semilla: las mismas muestras para cada umbral
            trace = HeatTrace(pyr, thr=thr, window_levels=(2, 3))
            # un solo lote, para que la traza cubra todas las muestras a la vez
            _, stats = generate(model, meta, labels, device, args.steps, batch=args.num, resolve_fresh=args.resolve,
                                revise=args.revise, revise_thr=args.revise_thr, trace=trace)
            s = trace.summary()
            res["umbrales"][str(thr)] = {"resumen": s, "por_paso": trace.steps}
            res["estadisticas"] = {k: round(float(v), 1) for k, v in stats.items()}
            w = s["ventanas_n2"]
            print(f"{path} | umbral {thr} | nodos calientes {s['nodos_calientes_pct']}% | "
                  f"tokens a recalcular con ventanas de ~{w['tokens_por_ventana']} tokens: {w['tokens_a_recalcular_pct']}% | "
                  f"cambios que se perderían: {w['cambios_fuera_de_ventanas_calientes_pct']}%", flush=True)
        name = os.path.basename(os.path.dirname(path)) + ("_resolve" if args.resolve else "") + \
            ("_sin_union" if args.revise <= 0 and meta["method"] == "grow_merge" else "")
        out = os.path.join(args.out_dir, f"heat_{name}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
        print("guardado", out, flush=True)


if __name__ == "__main__":
    main()
