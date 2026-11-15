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

from data.voxelize import fill_solid, voxelize_surface

STATE_MIXED = 2


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


def merge_hists(hists):
    total = {}
    for h in hists:
        for c, k in h.items():
            total[int(c)] = total.get(int(c), 0) + int(k)
    return total


def _H(counts):
    p = np.array(list(counts.values()), float)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def token_pieces(hist, d):
    """Descompone el token del omnitree en tres piezas y mide cada una (entropía de orden 0, en bits):
    corte (qué ejes se parten), ocupación (qué hijos tienen contenido) y lleno (cuáles de esos hijos son
    cajas llenas en vez de seguir subdividiéndose). Las dos últimas, condicionadas a las anteriores."""
    masks = cut_options("omni", d)
    cut, cut_occ, cut_occ_full, occ_only, full_only = {}, {}, {}, {}, {}
    for code, k in hist.items():
        ci, tern = divmod(int(code), 10000)
        nchild = 2 ** sum(masks[ci])
        digits = [(tern // 3 ** j) % 3 for j in range(nchild)]
        occ = sum(1 << j for j, s in enumerate(digits) if s != 0)
        full = sum(1 << j for j, s in enumerate(digits) if s == 1)
        for table, key in ((cut, ci), (cut_occ, (ci, occ)), (cut_occ_full, (ci, occ, full)),
                           (occ_only, (nchild, occ)), (full_only, (nchild, full))):
            table[key] = table.get(key, 0) + k
    h_cut, h_co, h_cof = _H(cut), _H(cut_occ), _H(cut_occ_full)
    return {"bits_corte": h_cut, "bits_ocupacion": h_co - h_cut, "bits_lleno": h_cof - h_co, "bits_total": h_cof,
            "simbolos_corte": len(cut), "simbolos_ocupacion": len(occ_only), "simbolos_lleno": len(full_only),
            "simbolos_conjuntos": len(cut_occ_full),
            "reparto_cortes": {"".join(map(str, masks[ci])): round(100 * k / sum(cut.values()), 1)
                               for ci, k in sorted(cut.items())}}


def tile_stats(vox):
    """Referencia de la bibliografía (OctLLM, JoVox, nuestro `tile`): un token por celda con contenido en los
    niveles por encima del más fino; su valor es qué hijos tienen contenido (un byte en 3D)."""
    n, S, d = vox.shape[0], vox.shape[1], vox.ndim - 1
    D = int(np.log2(S))
    cur, tokens, per_level, hist = vox.astype(bool), np.zeros(n), [], {}
    for _ in range(D):
        split = []
        for _ax in range(d):
            split += [cur.shape[1] // 2, 2]
        blocks = cur.reshape((n,) + tuple(split))
        order = [0] + [1 + 2 * i for i in range(d)] + [2 + 2 * i for i in range(d)]
        blocks = blocks.transpose(order).reshape((n,) + (cur.shape[1] // 2,) * d + (-1,))
        parent = blocks.any(-1)
        byte = (blocks * (1 << np.arange(blocks.shape[-1]))).sum(-1)[parent]
        for c, k in zip(*np.unique(byte, return_counts=True)):
            hist[int(c)] = hist.get(int(c), 0) + int(k)
        cnt = parent.reshape(n, -1).sum(1)
        tokens += cnt
        per_level.append(float(cnt.mean()))
        cur = parent
    return {"tokens": tokens, "hist": hist, "tokens_por_nivel": per_level[::-1]}


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
        vox = voxelize_surface(v, m["f"], args.res, points=1_500_000)
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
            hist = merge_hists([q["hist"] for q in parts])
            per_level = np.sum([np.array(q["tokens_por_nivel"]) * len(q["tokens"]) for q in parts], 0) / len(tok)
            results[f"{occ_name}_{kind}"] = {"tokens": tok.tolist(), "prof": prof.tolist(), "hojas": hojas.tolist(),
                                             "tokens_por_nivel": np.round(per_level, 1).tolist(),
                                             "hist": {str(c): k for c, k in hist.items()}}
            print(f"{occ_name:10s} {kind:4s} tokens {tok.mean():8.1f} (mediana {np.median(tok):.0f}, p95 "
                  f"{np.percentile(tok, 95):.0f}) | hojas {hojas.mean():8.1f} | prof. {prof.mean():.1f} (máx "
                  f"{prof.max():.0f}) | ~{bits:.2f} bits/token -> {bits * tok.mean():.0f} bits/forma | cortes de "
                  f"1/2/3 ejes: {np.round(100 * cuts / cuts.sum(), 1).tolist()} %", flush=True)
            if kind == "omni":
                pc = token_pieces(hist, occ.ndim - 1)
                results[f"{occ_name}_{kind}"]["piezas"] = pc
                print(f"{'':15s} token en 3 piezas: corte {pc['bits_corte']:.2f} + ocupación {pc['bits_ocupacion']:.2f} "
                      f"+ lleno {pc['bits_lleno']:.2f} = {pc['bits_total']:.2f} bits | símbolos por pieza: "
                      f"{pc['simbolos_corte']} / {pc['simbolos_ocupacion']} / {pc['simbolos_lleno']} "
                      f"(juntos: {pc['simbolos_conjuntos']}) | máx. tokens {tok.max():.0f}", flush=True)
                print(f"{'':15s} tokens por nivel: {np.round(per_level, 0).astype(int).tolist()}", flush=True)
        ts = [tile_stats(occ[i:i + args.batch]) for i in range(0, len(occ), args.batch)]
        tt = np.concatenate([q["tokens"] for q in ts])
        tb = _H(merge_hists([q["hist"] for q in ts]))
        lv = np.sum([np.array(q["tokens_por_nivel"]) * len(q["tokens"]) for q in ts], 0) / len(tt)
        results[f"{occ_name}_tile"] = {"tokens": tt.tolist(), "bits_token": tb, "tokens_por_nivel": lv.tolist()}
        print(f"{occ_name:10s} tile tokens {tt.mean():8.1f} (mediana {np.median(tt):.0f}, p95 {np.percentile(tt, 95):.0f}, "
              f"máx {tt.max():.0f}) | {tb:.2f} bits/token -> {tb * tt.mean():.0f} bits/forma | por nivel: "
              f"{np.round(lv, 0).astype(int).tolist()}", flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f)


if __name__ == "__main__":
    main()
