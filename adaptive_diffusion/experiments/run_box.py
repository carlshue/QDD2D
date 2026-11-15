"""
Un experimento 3D completo en un comando: entrena y evalúa, y deja todo en results/<nombre>/.

Uso (desde adaptive_diffusion/):
    python -m experiments.run_box --name box_omni_s0 --tree omni
    python -m experiments.run_box --name box_tile_s0 --tree tile
    python -m experiments.run_box --name prueba --tree omni --steps 300 --limit 500 --num 200   # prueba rápida

Los argumentos que no reconoce se pasan tal cual a experiments.train_box (--seed, --layers, --batch...).
Genera: train.log, eval.log, metrics_box.json, model.pt y env.json.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time


def run(cmd, log_path):
    print(">>", " ".join(cmd), flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        if proc.wait() != 0:
            raise SystemExit(f"El comando falló (código {proc.returncode}). Mira {log_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--tree", choices=["tile", "oct", "omni"], required=True)
    p.add_argument("--num", type=int, default=2000, help="muestras para evaluar")
    p.add_argument("--temps", default="1.0,0.9,0.8")
    p.add_argument("--skip_train", action="store_true", help="sólo evaluar un modelo ya entrenado")
    args, extra = p.parse_known_args()

    import torch
    out = os.path.join("results", args.name)
    os.makedirs(out, exist_ok=True)
    cuda = torch.cuda.is_available()
    env = {"fecha": time.strftime("%Y-%m-%d %H:%M:%S"), "sistema": platform.platform(),
           "python": platform.python_version(), "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0) if cuda else None, "args": vars(args), "extra": extra}
    try:
        env["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        env["commit"] = None
    print(f"GPU detectada: {env['gpu']}" if cuda else "AVISO: PyTorch no detecta ninguna GPU (muy lento).", flush=True)

    py = [sys.executable, "-u", "-m"]
    if not args.skip_train:
        t0 = time.time()
        run(py + ["experiments.train_box", "--tree", args.tree, "--out", out] + extra, os.path.join(out, "train.log"))
        env["minutos_entrenamiento"] = round((time.time() - t0) / 60, 1)
    run(py + ["experiments.eval_box", "--ckpt", os.path.join(out, "model.pt"), "--num", str(args.num),
              "--temps", args.temps], os.path.join(out, "eval.log"))
    with open(os.path.join(out, "env.json"), "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2, ensure_ascii=False)
    print(f"\nListo. Resultados en {out}")


if __name__ == "__main__":
    main()
