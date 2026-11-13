"""
Análisis sólo de muestreo sobre los modelos ya entrenados (no reentrena nada).
Responde a: ¿cuánto pesa el árbol aleatorio en el resultado de la difusión por
edición, y cuánto se debe a cómo nacen los nodos nuevos al muestrear?

  referencia   Fréchet y acierto de imágenes reales (suelo de la métrica)
  prior_solo   qué da el ruido sin red
  base         los cuatro modelos tal cual, 3 semillas de muestreo
  cruzado      modelo entrenado con el ruido A, muestreado con árbol inicial B
               y nodos nuevos C
  pasos        nº de pasos, con y sin resolver los nodos recién nacidos
  arbol_inicial  parecido entre muestras que comparten árbol de partida

Uso (desde adaptive_diffusion/, con GPU; unos minutos):
    python -m experiments.sampling_analysis
Escribe results/analysis/sampling_analysis.json
"""
import argparse
import json
import os
import numpy as np
import torch

from data.binary_mnist import load_binary_mnist
from experiments.common import generate, load_model, pick_device
from experiments.evaluate import embed, frechet, get_classifier
from models.edit_diffusion import sample
from structures.pyramid import Pyramid

EDIT = ("edit_cell", "edit_level", "edit_uniform")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=2000)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="results/analysis/sampling_analysis.json")
    args = p.parse_args()
    dev, n = pick_device(args.device), args.num
    amp = str(dev).startswith("cuda")

    clf = get_classifier(dev)
    test, _ = load_binary_mnist(train=False)
    train, train_y = load_binary_mnist(train=True)
    f_test, _ = embed(clf, test, dev)
    labels = torch.arange(n) % 10
    out = {}

    def score(imgs, labs=labels):
        feats, preds = embed(clf, imgs, dev)
        return {"acierto_clase_pct": round(100 * float((preds == labs.numpy()).mean()), 2),
                "frechet": round(frechet(feats, f_test), 2)}

    pyr = Pyramid(5, dev)
    cpu_pyr = Pyramid(5, "cpu")

    def canonical_nodes(imgs):
        s = cpu_pyr.from_images(torch.from_numpy(np.asarray(imgs)))
        return round(float(cpu_pyr.exists(s).sum(1).float().mean()), 1)

    def run_edit(model, start, fresh, seed=0, steps=32, resolve=False):
        g = torch.Generator(dev).manual_seed(seed)
        imgs, agg = [], []
        for lab in labels.split(500):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                im, st = sample(model, pyr, start, len(lab), steps, lab.to(dev), generator=g,
                                resolve_fresh=resolve, fresh_prior=fresh)
            imgs.append(im.cpu().numpy().astype(np.uint8)); agg.append(st)
        imgs = np.concatenate(imgs)
        r = score(imgs)
        for k in ("evals", "tokens_por_muestra", "nodos_inicio", "nodos_final"):
            r[k] = round(float(np.mean([a[k] for a in agg])), 1)
        r["nodos_canonicos"] = canonical_nodes(imgs)
        return r

    def log(section, key, r):
        out.setdefault(section, {})[key] = r
        print(section, key, json.dumps(r, ensure_ascii=False), flush=True)

    models, priors = {}, {}
    for name in EDIT:
        models[name], meta = load_model(f"results/{name}/model.pt", dev)
        priors[name] = meta["prior"].to(dev)
    grow, grow_meta = load_model("results/gpu01/model.pt", dev)

    # 0. suelo de la métrica: subconjuntos al azar de entrenamiento frente a prueba
    for s in range(3):
        idx = np.random.default_rng(s).choice(len(train), n, replace=False)
        log("referencia", f"reales_s{s}", score(train[idx], torch.from_numpy(train_y[idx])))

    # 1. el ruido por sí solo
    s0 = pyr.from_images(torch.from_numpy(train).to(dev))
    ex0 = pyr.exists(s0)
    for name in EDIT:
        st = pyr.sample_states(priors[name], n, torch.Generator(dev).manual_seed(0))
        ex = pyr.exists(st)
        imgs = pyr.to_image(st, ex).cpu().numpy().astype(np.uint8)
        r = score(imgs)
        r["nodos"] = round(ex.sum(1).float().mean().item(), 1)
        r["pixeles_llenos_pct"] = round(100 * float(imgs.mean()), 2)
        # prob. de que un nodo real corrompido coincida por azar con su estado verdadero
        p_true = priors[name][None].expand(1000, -1, -1).gather(2, s0[:1000].long()[..., None])[..., 0]
        r["coincide_por_azar_pct"] = round(100 * float((p_true * ex0[:1000]).sum() / ex0[:1000].sum()), 1)
        log("prior_solo", name, r)
    out["datos"] = {"nodos": round(ex0.sum(1).float().mean().item(), 1),
                    "pixeles_llenos_pct": round(100 * float(train.mean()), 2)}

    # 2. los cuatro modelos tal cual
    for s in range(3):
        torch.manual_seed(s)
        imgs, st = generate(grow, grow_meta, labels, dev)
        r = score(imgs); r.update({k: round(float(v), 1) for k, v in st.items()})
        r["nodos_canonicos"] = canonical_nodes(imgs)
        log("base", f"grow_s{s}", r)
    for name in EDIT:
        for s in range(3):
            log("base", f"{name}_s{s}", run_edit(models[name], priors[name], priors[name], seed=s))

    # 3. modelo entrenado con A, muestreado con árbol inicial B y nodos nuevos C
    for m in EDIT:
        for a in EDIT:
            for b in EDIT:
                log("cruzado", f"{m}|inicio={a}|nuevos={b}", run_edit(models[m], priors[a], priors[b]))

    # 4. pasos, con y sin resolver los nodos recién nacidos
    for name in EDIT:
        for steps in (8, 16, 64, 128):
            log("pasos", f"{name}|{steps}", run_edit(models[name], priors[name], priors[name], steps=steps))
        for steps in (6, 8, 16, 32):
            log("pasos", f"{name}|{steps}|resolve",
                run_edit(models[name], priors[name], priors[name], steps=steps, resolve=True))
    for s in (1, 2):
        log("pasos", f"edit_cell|32|resolve|s{s}",
            run_edit(models["edit_cell"], priors["edit_cell"], priors["edit_cell"], seed=s, resolve=True))
    for spl in (4, 16):
        torch.manual_seed(0)
        imgs, st = generate(grow, grow_meta, labels, dev, steps=spl)
        r = score(imgs); r.update({k: round(float(v), 1) for k, v in st.items()})
        log("pasos", f"grow|{spl}_por_nivel", r)

    # 5. ¿la imagen final depende del árbol de partida concreto?
    m_n = 1000
    lab = (torch.arange(m_n) % 10).to(dev)

    def iou(a, b):
        return round(float(((a * b).sum((1, 2)) / (a + b).clamp(max=1).sum((1, 2)).clamp_min(1)).mean()), 3)

    for name in EDIT:
        g0 = torch.Generator(dev).manual_seed(123)
        init_a = pyr.sample_states(priors[name], m_n, g0)
        init_b = pyr.sample_states(priors[name], m_n, g0)
        res = {}
        for tag, init, seed in (("a1", init_a, 1), ("a2", init_a, 2), ("b", init_b, 3)):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                im, _ = sample(models[name], pyr, priors[name], m_n, 32, lab, init_state=init,
                               generator=torch.Generator(dev).manual_seed(seed))
            res[tag] = im.float()
        start_img = pyr.to_image(init_a, pyr.exists(init_a)).float()
        log("arbol_inicial", name, {"iou_mismo_arbol": iou(res["a1"], res["a2"]),
                                    "iou_distinto_arbol": iou(res["a1"], res["b"]),
                                    "iou_inicio_con_su_final": iou(start_img, res["a1"]),
                                    "iou_inicio_con_otro_final": iou(start_img, res["b"])})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("guardado", args.out)


if __name__ == "__main__":
    main()
