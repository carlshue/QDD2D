"""
Tanda nocturna: lanza varios experimentos seguidos, evalúa gpu01 como
referencia, sube los resultados a git y apaga el ordenador.

Uso (desde adaptive_diffusion/):
    python -m experiments.run_night
    python -m experiments.run_night --no_shutdown      # no apagar al terminar
    python -m experiments.run_night --dry_run          # sólo mostrar lo que haría

- Un experimento que ya tiene metrics.json se salta (ya está terminado).
- Si uno falla, se anota y se sigue con el siguiente.
- El apagado espera 2 minutos; se cancela con:  shutdown /a   (Windows)
"""
import argparse
import os
import platform
import subprocess
import sys
import time

EXPERIMENTS = [("edit_cell", "cell"), ("edit_level", "level"), ("edit_uniform", "uniform")]
LOG = os.path.join("results", "night.log")


def note(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs("results", exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd, dry):
    note(">> " + " ".join(cmd))
    if dry:
        return True
    try:
        return subprocess.call(cmd) == 0
    except Exception as e:  # p. ej. git no encontrado
        note(f"error al lanzar: {e}")
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--no_shutdown", action="store_true")
    p.add_argument("--no_push", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    py = [sys.executable, "-u", "-m"]
    note(f"inicio de la tanda ({'simulacro' if args.dry_run else 'real'})")

    for name, prior in EXPERIMENTS:
        if os.path.exists(os.path.join("results", name, "metrics.json")):
            note(f"{name}: ya terminado, se salta")
            continue
        t0 = time.time()
        ok = run(py + ["experiments.run_local", "--name", name, "--method", "edit", "--prior", prior,
                       "--steps", str(args.steps)], args.dry_run)
        note(f"{name}: {'correcto' if ok else 'FALLÓ'} en {(time.time() - t0) / 60:.0f} min")

    ref = os.path.join("results", "gpu01", "model.pt")
    if os.path.exists(ref) and not os.path.exists(os.path.join("results", "gpu01", "metrics.json")):
        ok = run(py + ["experiments.evaluate", "--ckpt", ref], args.dry_run)
        note(f"evaluación de gpu01: {'correcta' if ok else 'FALLÓ'}")

    if not args.no_push:
        note("subiendo resultados a git")
        run(["git", "add", "results"], args.dry_run)
        run(["git", "commit", "-m", "Resultados de la tanda nocturna (edit: cell, level, uniform)"], args.dry_run)
        pushed = run(["git", "push"], args.dry_run)
        note("push correcto" if pushed else "el push FALLÓ: los resultados quedan en el disco, en results/")

    if args.no_shutdown:
        note("fin de la tanda (sin apagar)")
        return
    note("fin de la tanda: el ordenador se apaga en 2 minutos")
    if platform.system() == "Windows":
        run(["shutdown", "/s", "/t", "120"], args.dry_run)
    else:
        run(["shutdown", "-h", "+2"], args.dry_run)


if __name__ == "__main__":
    main()
