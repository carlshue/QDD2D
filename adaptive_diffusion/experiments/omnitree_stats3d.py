"""
Octree frente a omnitree frente a bintree en 3D, sobre mallas voxelizadas.

Para cada forma se calcula, por programación dinámica sobre todas las cajas diádicas, el árbol exacto
(sin pérdida) con menos nodos internos de cada tipo:
  oct  : cada nodo mixto parte los tres ejes (8 hijos)
  omni : cada nodo mixto elige qué ejes partir (2, 4 u 8 hijos)
  bin  : cada nodo mixto parte un solo eje (2 hijos)
Un "token" = un nodo interno (mixto): etiqueta de corte + estado de cada hijo {vacío, lleno, mixto}.

Dos ocupaciones por forma:
  superficie : vóxel ocupado si la superficie pasa por él (lo que generan OctGPT o JoVox)
  sólido     : además, el interior relleno (sólo cambia en mallas cerradas)

Entrada: una carpeta de ficheros .npz con `v` (vértices, N×3) y `f` (caras, M×3).

Uso (desde adaptive_diffusion/):
    python -m experiments.omnitree_stats3d --meshes <carpeta> --res 64 --num 300
"""
import argparse
import glob
import itertools
import json
import os
import numpy as np

STATE_MIXED = 2


def voxelize_surface(v, f, res, points=1_500_000, rng=None):
    """Ocupación de superficie (res³, bool): muestrea puntos en los triángulos y los reparte en vóxeles."""
    rng = rng or np.random.default_rng(0)
    lo, hi = v.min(0), v.max(0)
    v = (v - (lo + hi) / 2) / (hi - lo).max() * 0.98 + 0.5          # caja unidad, lado mayor = 0,98
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    if not np.isfinite(area).all() or area.sum() <= 0:
        return None
    tri = rng.choice(len(f), points, p=area / area.sum())
    u, w = rng.random(points), rng.random(points)
    flip = u + w > 1
    u[flip], w[flip] = 1 - u[flip], 1 - w[flip]
    p = a[tri] + u[:, None] * (b[tri] - a[tri]) + w[:, None] * (c[tri] - a[tri])
    idx = np.clip((p * res).astype(np.int64), 0, res - 1)
    vox = np.zeros((res, res, res), bool)
    vox[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return vox


def fill_solid(surface):
    """Rellena el interior: todo lo que no se alcanza desde el borde por celdas vacías."""
    free = ~surface
    out = np.zeros_like(surface)
    out[0], out[-1], out[:, 0], out[:, -1], out[:, :, 0], out[:, :, -1] = True, True, True, True, True, True
    out &= free
    while True:
        grow = out.copy()
        grow[1:] |= out[:-1]; grow[:-1] |= out[1:]
        grow[:, 1:] |= out[:, :-1]; grow[:, :-1] |= out[:, 1:]
        grow[:, :, 1:] |= out[:, :, :-1]; grow[:, :, :-1] |= out[:, :, 1:]
        grow &= free
        if (grow == out).all():
            return ~out
        out = grow


def cut_options(kind, d):
    masks = [m for m in itertools.product((0, 1), repeat=d) if any(m)]
    if kind == "oct":
        masks = [m for m in masks if all(m)]
    elif kind == "bin":
        masks = [m for m in masks if sum(m) == 1]
    return sorted(masks, key=lambda m: (-sum(m), m))      # desempate: partir más ejes (árbol menos profundo)


def analyse(vox, kind):
    """vox: (n, S, ..., S) en {0,1}, d ejes espaciales. Devuelve estadísticas del árbol óptimo."""
    n, S, d = vox.shape[0], vox.shape[1], vox.ndim - 1
    D = int(np.log2(S))
    levels = list(itertools.product(range(D + 1), repeat=d))
    add = lambda lv, m: tuple(a + b for a, b in zip(lv, m))
    shape = lambda lv: (n,) + tuple(2 ** l for l in lv)

    def kids(arr, lv, m):
        """Valores de los hijos de cada nodo del nivel lv con el corte m: (n, *shape(lv), 2^|m|)."""
        a = arr[add(lv, m)]
        split = []
        for l, b in zip(lv, m):
            split += [2 ** l, 2 ** b]
        a = a.reshape((n,) + tuple(split))
        order = [0] + [1 + 2 * i for i in range(d)] + [2 + 2 * i for i in range(d)]
        return a.transpose(order).reshape(shape(lv) + (-1,))

    # uniformidad y valor de cada caja diádica, de fino a grueso
    mx, mn = {(D,) * d: vox.astype(np.uint8)}, {(D,) * d: vox.astype(np.uint8)}
    for lv in sorted(levels, key=lambda t: -sum(t)):
        if lv in mx:
            continue
        j = next(i for i in range(d) if lv[i] < D)
        m = tuple(int(i == j) for i in range(d))
        mx[lv], mn[lv] = kids(mx, lv, m).max(-1), kids(mn, lv, m).min(-1)
    uni = {lv: mx[lv] == mn[lv] for lv in levels}
    state = {lv: np.where(uni[lv], mx[lv], STATE_MIXED).astype(np.int32) for lv in levels}
    del mn

    all_opts = cut_options(kind, d)
    opt_id = {m: i for i, m in enumerate(cut_options("omni", d))}
    cost, depth, leaves, choice = {}, {}, {}, {}
    for lv in sorted(levels, key=lambda t: -sum(t)):
        opts = [m for m in all_opts if all(l + b <= D for l, b in zip(lv, m))]
        u = uni[lv]
        if not opts:
            cost[lv] = np.where(u, 0, 10 ** 7).astype(np.int32)
            depth[lv] = np.zeros(u.shape, np.int16); leaves[lv] = np.ones(u.shape, np.int32)
            choice[lv] = np.full(u.shape, -1, np.int8)
            continue
        cs = np.stack([kids(cost, lv, m).sum(-1) for m in opts])
        best = cs.argmin(0)
        pick = lambda stack: np.take_along_axis(stack, best[None], 0)[0]
        cost[lv] = np.where(u, 0, 1 + pick(cs)).astype(np.int32)
        depth[lv] = np.where(u, 0, 1 + pick(np.stack([kids(depth, lv, m).max(-1) for m in opts]))).astype(np.int16)
        leaves[lv] = np.where(u, 1, pick(np.stack([kids(leaves, lv, m).sum(-1) for m in opts]))).astype(np.int32)
        choice[lv] = np.where(u, -1, np.array([opt_id[m] for m in opts])[best]).astype(np.int8)

    # recorrer el árbol elegido desde la raíz para contar los tokens reales
    root = (0,) * d
    reach = {lv: np.zeros(shape(lv), bool) for lv in levels}
    reach[root][:] = True
    lvl = {lv: np.zeros(shape(lv), np.int16) for lv in levels}
    hist, cuts, per_depth = {}, np.zeros(d + 1, np.int64), np.zeros(d * D + 1)
    for lv in sorted(levels, key=sum):
        node = reach[lv] & ~uni[lv]
        if not node.any():
            continue
        for m, ci in opt_id.items():
            sel = node & (choice[lv] == ci)
            if not sel.any():
                continue
            cuts[sum(m)] += sel.sum()
            st = kids(state, lv, m)[sel]
            code = (st * 3 ** np.arange(st.shape[-1])).sum(-1) + 10000 * ci
            for c, k in zip(*np.unique(code, return_counts=True)):
                hist[c] = hist.get(c, 0) + k
            np.add.at(per_depth, lvl[lv][sel], 1)
            rep, dep = sel, lvl[lv] * sel
            for ax, b in enumerate(m):
                if b:
                    rep, dep = np.repeat(rep, 2, ax + 1), np.repeat(dep, 2, ax + 1)
            child = add(lv, m)
            reach[child] |= rep
            lvl[child] = np.where(rep, dep + 1, lvl[child])
    tok = cost[root].reshape(n)
    return {"tokens": tok.astype(float), "hojas": leaves[root].reshape(n).astype(float),
            "prof": depth[root].reshape(n).astype(float), "hist": hist,
            "cortes_por_n_ejes": cuts[1:].tolist(), "tokens_por_nivel": (per_depth / n).tolist()}


def entropy_bits(hists):
    """Entropía de orden 0 (bits por token) del conjunto de símbolos de varios lotes."""
    total = {}
    for h in hists:
        for c, k in h.items():
            total[c] = total.get(c, 0) + k
    p = np.array(list(total.values()), float)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--meshes", required=True, help="carpeta con .npz (v, f)")
    p.add_argument("--res", type=int, default=64)
    p.add_argument("--num", type=int, default=300)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--kinds", default="oct,omni,bin")
    p.add_argument("--rotate", type=float, default=0.0, help="grados de giro alrededor del eje vertical (Y)")
    p.add_argument("--out", default=None, help="json con los resultados por forma")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.meshes, "*.npz")))[:args.num]
    names, surf, solid = [], [], []
    for path in files:
        m = np.load(path)
        v = m["v"].astype(np.float64)
        if args.rotate:
            t = np.deg2rad(args.rotate)
            v = v @ np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]]).T
        vox = voxelize_surface(v, m["f"], args.res)
        if vox is None:
            continue
        names.append(os.path.splitext(os.path.basename(path))[0])
        surf.append(vox); solid.append(fill_solid(vox))
    surf, solid = np.stack(surf), np.stack(solid)
    closed = solid.reshape(len(solid), -1).sum(1) > 1.05 * surf.reshape(len(surf), -1).sum(1)
    print(f"{len(names)} formas a {args.res}³ | vóxeles de superficie: {surf.reshape(len(surf), -1).sum(1).mean():.0f} "
          f"de media | con interior relleno: {closed.mean() * 100:.0f} %", flush=True)

    results = {"names": names, "closed": closed.tolist()}
    for occ_name, occ in (("superficie", surf), ("solido", solid)):
        for kind in args.kinds.split(","):
            parts = [analyse(occ[i:i + args.batch], kind) for i in range(0, len(occ), args.batch)]
            tok = np.concatenate([q["tokens"] for q in parts])
            prof = np.concatenate([q["prof"] for q in parts])
            hojas = np.concatenate([q["hojas"] for q in parts])
            bits = entropy_bits([q["hist"] for q in parts])
            cuts = np.sum([q["cortes_por_n_ejes"] for q in parts], 0)
            results[f"{occ_name}_{kind}"] = {"tokens": tok.tolist(), "prof": prof.tolist(), "hojas": hojas.tolist()}
            print(f"{occ_name:10s} {kind:4s} tokens {tok.mean():8.1f} (mediana {np.median(tok):.0f}, p95 "
                  f"{np.percentile(tok, 95):.0f}) | hojas {hojas.mean():8.1f} | prof. {prof.mean():.1f} (máx "
                  f"{prof.max():.0f}) | ~{bits:.2f} bits/token -> {bits * tok.mean():.0f} bits/forma | cortes de "
                  f"1/2/3 ejes: {np.round(100 * cuts / cuts.sum(), 1).tolist()} %", flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f)


if __name__ == "__main__":
    main()
