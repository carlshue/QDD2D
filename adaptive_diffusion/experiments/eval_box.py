"""
Evalúa modelos 3D (octree u omnitree) generando formas y comparándolas con ABO.

  acierto de clase  % de muestras que un clasificador 3D reconoce como la clase pedida
  Fréchet           distancia entre características de muestras y de formas reales (menor es mejor)
  1-NNA             vecino más cercano entre muestras y formas de prueba: 50 % es lo ideal, 100 % lo peor
  cobertura         % de formas de prueba que son el vecino más cercano de alguna muestra
  MMD               distancia media de cada forma de prueba a su muestra más cercana
  copia             parecido medio (IoU) de cada muestra con la forma de entrenamiento más cercana;
                    se compara con el mismo número para las formas de prueba
  coste             evaluaciones de red y tokens procesados por muestra

La distancia entre formas es 1 - IoU de sus vóxeles.

Uso (desde adaptive_diffusion/):
    python -m experiments.eval_box --ckpt results/box_omni/model.pt results/box_tile/model.pt --temps 1.0,0.9
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.abo import CLASSES
from experiments.box_common import load_model, load_shapes
from experiments.common import pick_device
from experiments.evaluate import frechet
from models.box_grow import augment, sample
from structures.boxtree import tokens_to_voxels


class VoxelClassifier(nn.Module):
    def __init__(self, res, classes=len(CLASSES)):
        super().__init__()
        self.c1, self.c2, self.c3 = nn.Conv3d(1, 16, 3, padding=1), nn.Conv3d(16, 32, 3, padding=1), nn.Conv3d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * (res // 8) ** 3, 64)
        self.out = nn.Linear(64, classes)

    def features(self, x):
        for conv in (self.c1, self.c2, self.c3):
            x = F.max_pool3d(F.relu(conv(x)), 2)
        return F.relu(self.fc(x.flatten(1)))

    def forward(self, x):
        return self.out(self.features(x))


def get_classifier(res, train_vox, train_y, device, epochs=20):
    path = os.path.join("results", f"classifier_abo{res}.pt")
    clf = VoxelClassifier(res).to(device)
    if os.path.exists(path):
        clf.load_state_dict(torch.load(path, map_location=device))
        return clf.eval()
    print("entrenando el clasificador 3D de evaluación (una sola vez)...", flush=True)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    opt = torch.optim.Adam(clf.parameters(), 1e-3)
    y = torch.from_numpy(train_y)
    for _ in range(epochs):
        perm = rng.permutation(len(train_vox))
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            x = torch.from_numpy(augment(train_vox[idx], int(rng.integers(8)))).float()[:, None].to(device)
            loss = F.cross_entropy(clf(x), y[idx].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    os.makedirs("results", exist_ok=True)
    torch.save(clf.state_dict(), path)
    return clf.eval()


@torch.no_grad()
def embed(clf, vox, device):
    feats, preds = [], []
    for i in range(0, len(vox), 256):
        f = clf.features(torch.from_numpy(vox[i:i + 256]).float()[:, None].to(device))
        feats.append(f.double().cpu())
        preds.append(clf.out(f).argmax(1).cpu())
    return torch.cat(feats).numpy(), torch.cat(preds).numpy()


@torch.no_grad()
def iou_matrix(a, b, device, chunk=256):
    """IoU entre cada rejilla de a (n, ...) y cada una de b (m, ...). Devuelve (n, m) float32 en CPU."""
    fb = torch.from_numpy(b.reshape(len(b), -1)).float().to(device)
    sb = fb.sum(1)
    out = []
    for i in range(0, len(a), chunk):
        fa = torch.from_numpy(a[i:i + chunk].reshape(-1, fb.shape[1])).float().to(device)
        inter = fa @ fb.T
        out.append((inter / (fa.sum(1)[:, None] + sb[None] - inter).clamp_min(1)).cpu())
    return torch.cat(out)


def set_metrics(gen, ref, device):
    """1-NNA, cobertura y MMD entre dos conjuntos del mismo tamaño, con distancia 1 - IoU."""
    both = np.concatenate([gen, ref])
    d = 1 - iou_matrix(both, both, device)
    d.fill_diagonal_(float("inf"))
    n = len(gen)
    nn_is_gen = d.argmin(1) < n
    nna = float(torch.cat([nn_is_gen[:n], ~nn_is_gen[n:]]).float().mean())
    gr = d[:n, n:]
    return {"nna_pct": 100 * nna, "cobertura_pct": 100 * len(torch.unique(gr.argmin(1))) / len(ref),
            "mmd": float(gr.min(0).values.mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--num", type=int, default=2000)
    p.add_argument("--temps", default="1.0,0.9,0.8")
    p.add_argument("--seeds", default="0")
    p.add_argument("--steps", type=int, default=8, help="pasos por nivel del árbol")
    p.add_argument("--batch", type=int, default=100)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device = pick_device(args.device)
    use_amp = device.startswith("cuda")

    for path in args.ckpt:
        model, a = load_model(path, device)
        res, cap = a["res"], a["max_tile_tokens"]
        train, train_y = load_shapes(res, True, cap)
        test, test_y = load_shapes(res, False, cap)
        clf = get_classifier(res, train, train_y, device)
        f_train, _ = embed(clf, train, device)
        f_test, pred_test = embed(clf, test, device)
        copy_test = float(iou_matrix(test, train, device).max(1).values.mean())
        print(f"\n== {path} (árbol '{a['tree']}', {res}³) | datos reales: acierto {100 * (pred_test == test_y).mean():.1f} %, "
              f"Fréchet prueba-entrenamiento {frechet(f_test, f_train):.2f} (suelo), copia {copy_test:.3f}", flush=True)
        results = []
        for seed in map(int, args.seeds.split(",")):
            for temp in map(float, args.temps.split(",")):
                torch.manual_seed(seed)
                labels = torch.from_numpy(np.random.default_rng(seed).choice(train_y, args.num))
                trees, agg, t0 = [], {}, time.time()
                for i in range(0, args.num, args.batch):
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                        out, st = sample(model, len(labels[i:i + args.batch]), args.steps, labels[i:i + args.batch],
                                         temp, device, max_tokens=4 * (cap or 4096))
                    trees += out
                    for k, v in st.items():
                        agg.setdefault(k, []).append(v)
                secs = (time.time() - t0) / args.num
                vox = np.stack([tokens_to_voxels(t, model.depth) for t in trees])
                feats, preds = embed(clf, vox, device)
                m = min(len(test), args.num)
                r = {"modelo": path, "arbol": a["tree"], "temperatura": temp, "semilla": seed, "muestras": args.num,
                     "acierto_clase_pct": round(100 * float((preds == labels.numpy()).mean()), 2),
                     "frechet": round(frechet(feats, f_train), 2),
                     **{k: round(v, 3) for k, v in set_metrics(vox[:m], test[:m], device).items()},
                     "copia": round(float(iou_matrix(vox, train, device).max(1).values.mean()), 3),
                     "copia_prueba": round(copy_test, 3), "vacias": int((vox.reshape(len(vox), -1).sum(1) == 0).sum()),
                     "voxeles": round(float(vox.reshape(len(vox), -1).sum(1).mean()), 1),
                     "evals": round(float(np.mean(agg["evals"])), 1),
                     "tokens_por_muestra": round(float(np.mean(agg["tokens_por_muestra"])), 1),
                     "nodos": round(float(np.mean(agg["nodos"])), 1), "nodos_max": int(max(agg["nodos_max"])),
                     "profundidad": round(float(np.mean(agg["profundidad"])), 2),
                     "truncados": int(sum(agg["truncados"])), "seg_por_muestra": round(secs, 4)}
                results.append(r)
                print(json.dumps(r, ensure_ascii=False), flush=True)
        with open(os.path.join(os.path.dirname(path), "metrics_box.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
