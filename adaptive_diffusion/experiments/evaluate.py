"""
Mide la calidad de uno o varios modelos con un clasificador de MNIST binario.

  acierto de clase : % de muestras que el clasificador reconoce como el dígito pedido
  distancia Fréchet: distancia entre las características de imágenes generadas y
                     reales (como el FID, pero con este clasificador; menor es mejor)
  coste            : evaluaciones de red y tokens procesados por muestra

Uso (desde adaptive_diffusion/):
    python -m experiments.evaluate --ckpt results/gpu01/model.pt results/edit01/model.pt
"""
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.binary_mnist import load_binary_mnist
from experiments.common import generate, load_model, pick_device


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.c3 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 4 * 4, 128)
        self.out = nn.Linear(128, 10)

    def features(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = F.max_pool2d(F.relu(self.c3(x)), 2)
        return F.relu(self.fc(x.flatten(1)))

    def forward(self, x):
        return self.out(self.features(x))


def get_classifier(device, path="results/classifier.pt", epochs=3):
    clf = Classifier().to(device)
    if os.path.exists(path):
        clf.load_state_dict(torch.load(path, map_location=device))
        return clf.eval()
    print("entrenando el clasificador de evaluación (una sola vez)...", flush=True)
    torch.manual_seed(0)
    imgs, labels = load_binary_mnist(train=True)
    x = torch.from_numpy(imgs).float()[:, None].to(device)
    y = torch.from_numpy(labels).to(device)
    opt = torch.optim.Adam(clf.parameters(), 1e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(x), device=device)
        for i in range(0, len(x), 256):
            idx = perm[i:i + 256]
            loss = F.cross_entropy(clf(x[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(clf.state_dict(), path)
    return clf.eval()


@torch.no_grad()
def embed(clf, imgs, device):
    x = torch.from_numpy(np.asarray(imgs)).float()[:, None]
    feats, preds = [], []
    for chunk in x.split(1000):
        f = clf.features(chunk.to(device))
        feats.append(f.double().cpu()); preds.append(clf.out(f).argmax(1).cpu())
    return torch.cat(feats).numpy(), torch.cat(preds).numpy()


def frechet(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    eig = np.linalg.eigvals(s1 @ s2)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.sqrt(np.clip(eig.real, 0, None)).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--num", type=int, default=2000, help="muestras por modelo")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = pick_device(args.device)
    clf = get_classifier(device)
    test, test_y = load_binary_mnist(train=False)
    train, _ = load_binary_mnist(train=True)
    f_test, pred_test = embed(clf, test, device)
    f_train, _ = embed(clf, train[:args.num], device)
    print(f"referencia con imágenes reales: acierto {100 * (pred_test == test_y).mean():.1f}% | "
          f"Fréchet entrenamiento-prueba {frechet(f_train, f_test):.2f} (suelo de la métrica)", flush=True)

    labels = torch.arange(args.num) % 10
    for path in args.ckpt:
        torch.manual_seed(args.seed)
        model, meta = load_model(path, device)
        t0 = time.time()
        imgs, stats = generate(model, meta, labels, device, args.steps, args.temperature)
        secs = time.time() - t0
        feats, preds = embed(clf, imgs, device)
        res = {"modelo": path, "metodo": meta["method"], "muestras": args.num,
               "acierto_clase_pct": round(100 * float((preds == labels.numpy()).mean()), 2),
               "frechet": round(frechet(feats, f_test), 2),
               "segundos_por_muestra": round(secs / args.num, 4),
               **{k: round(float(v), 1) for k, v in stats.items()}}
        print(json.dumps(res, ensure_ascii=False), flush=True)
        with open(os.path.join(os.path.dirname(path), "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
