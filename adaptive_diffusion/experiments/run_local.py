"""
Ejecuta un experimento completo en tu máquina y deja los resultados en
results/<nombre>/ listos para subir a git.

Uso (desde adaptive_diffusion/):
    python -m experiments.run_local --name edit_cell --method edit --prior cell
    python -m experiments.run_local --name grow02 --method grow

Genera: train.log, sample.log, samples.png, eval.log, metrics.json, model.pt y env.json.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time


def run(cmd, log_path):
    """Lanza un comando mostrando su salida y guardándola en log_path."""
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
    p.add_argument("--name", required=True, help="nombre del experimento, p. ej. gpu01")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--method", choices=["grow", "edit"], default="edit")
    p.add_argument("--prior", choices=["uniform", "level", "cell"], default="cell")
    p.add_argument("--eval_samples", type=int, default=2000)
    args = p.parse_args()

    import torch
    out = os.path.join("results", args.name)
    os.makedirs(out, exist_ok=True)
    cuda = torch.cuda.is_available()
    env = {
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sistema": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_memoria_gb": round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1) if cuda else None,
        "args": vars(args),
    }
    try:
        env["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        env["commit"] = None

    py = [sys.executable, "-u", "-m"]   # -u: salida sin búfer, para ver el progreso en directo
    train = py + ["experiments.run_training", "--out", out, "--steps", str(args.steps), "--batch", str(args.batch),
                  "--lr", str(args.lr), "--dim", str(args.dim), "--layers", str(args.layers),
                  "--heads", str(args.heads), "--device", args.device,
                  "--method", args.method, "--prior", args.prior]
    if args.limit:
        train += ["--limit", str(args.limit)]
    print(f"GPU detectada: {env['gpu']}" if cuda else
          "AVISO: PyTorch no detecta ninguna GPU; se entrenará en CPU (muy lento).", flush=True)
    t0 = time.time()
    run(train, os.path.join(out, "train.log"))
    env["minutos_entrenamiento"] = round((time.time() - t0) / 60, 1)
    ckpt = os.path.join(out, "model.pt")
    run(py + ["experiments.run_sampling", "--ckpt", ckpt, "--num", str(args.num_samples), "--device", args.device],
        os.path.join(out, "sample.log"))
    run(py + ["experiments.evaluate", "--ckpt", ckpt, "--num", str(args.eval_samples), "--device", args.device],
        os.path.join(out, "eval.log"))
    with open(os.path.join(out, "env.json"), "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2, ensure_ascii=False)

    print(f"\nListo. Resultados en {out}. Para subirlos:")
    print(f"  git add {out}")
    print(f'  git commit -m "Resultados {args.name}"')
    print("  git push")


if __name__ == "__main__":
    main()
