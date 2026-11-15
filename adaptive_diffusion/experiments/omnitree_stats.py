"""
Compara, sobre imágenes binarias, tres árboles exactos (sin pérdida):
  quad : quadtree (cada nodo mixto parte los dos ejes)
  omni : omnitree (cada nodo mixto elige partir x, y o ambos), óptimo en nº de nodos internos
  bin  : bintree (sólo un eje por nodo), óptimo
Un "token" = un nodo interno (mixto): etiqueta de corte + estado de cada hijo {vacío, lleno, mixto}.
El árbol óptimo se obtiene por programación dinámica sobre todos los rectángulos diádicos.

Uso (desde adaptive_diffusion/):  python -m experiments.omnitree_stats
"""
import numpy as np

from data.datasets import load_images

OPTS = {"quad": ("b",), "omni": ("b", "y", "x"), "bin": ("y", "x")}   # orden = desempate


def analyse(imgs, kind):
    n, S, _ = imgs.shape
    D = int(np.log2(S))
    uni, val = {}, {}
    for ly in range(D + 1):
        for lx in range(D + 1):
            b = imgs.reshape(n, 2 ** ly, S >> ly, 2 ** lx, S >> lx)
            mx, mn = b.max((2, 4)), b.min((2, 4))
            uni[ly, lx], val[ly, lx] = mx == mn, mx

    def kids(arr, ly, lx, o):          # suma (u otra reducción) de los hijos de cada nodo de (ly, lx)
        if o == "y":
            a = arr[ly + 1, lx]; return a.reshape(n, 2 ** ly, 2, 2 ** lx).transpose(0, 1, 3, 2)
        if o == "x":
            a = arr[ly, lx + 1]; return a.reshape(n, 2 ** ly, 2 ** lx, 2)
        a = arr[ly + 1, lx + 1]
        return a.reshape(n, 2 ** ly, 2, 2 ** lx, 2).transpose(0, 1, 3, 2, 4).reshape(n, 2 ** ly, 2 ** lx, 4)

    cost, depth, leaves, choice = {}, {}, {}, {}
    for ly in range(D, -1, -1):
        for lx in range(D, -1, -1):
            opts = [o for o in OPTS[kind] if (o != "y" or ly < D) and (o != "x" or lx < D)
                    and (o != "b" or (ly < D and lx < D))]
            if kind == "quad" and not opts and (ly < D or lx < D):
                opts = []                                    # rectángulos no cuadrados: no se usan
            u = uni[ly, lx]
            if not opts:
                cost[ly, lx] = np.where(u, 0, 10 ** 6); depth[ly, lx] = np.zeros_like(cost[ly, lx])
                leaves[ly, lx] = np.ones_like(cost[ly, lx]); choice[ly, lx] = np.full(u.shape, -1)
                continue
            cs = np.stack([kids(cost, ly, lx, o).sum(-1) for o in opts])
            best = cs.argmin(0)
            pick = lambda stack: np.take_along_axis(stack, best[None], 0)[0]
            cost[ly, lx] = np.where(u, 0, 1 + pick(cs))
            depth[ly, lx] = np.where(u, 0, 1 + pick(np.stack([kids(depth, ly, lx, o).max(-1) for o in opts])))
            leaves[ly, lx] = np.where(u, 1, pick(np.stack([kids(leaves, ly, lx, o).sum(-1) for o in opts])))
            choice[ly, lx] = np.where(u, -1, np.array(["byx".index(o) for o in opts])[best])

    # recorrer el árbol elegido desde la raíz para sacar los tokens reales
    reach = {k: np.zeros(v.shape, bool) for k, v in uni.items()}
    reach[0, 0][:] = True
    state = {k: np.where(uni[k], val[k], 2).astype(np.int64) for k in uni}
    hist, labels, per_depth = {}, np.zeros(3, np.int64), np.zeros(2 * D + 1)
    lvl = {k: np.zeros(v.shape, np.int64) for k, v in uni.items()}
    for s in range(2 * D + 1):
        for ly in range(D + 1):
            lx = s - ly
            if not 0 <= lx <= D:
                continue
            node = reach[ly, lx] & ~uni[ly, lx]
            for ci, o in enumerate("byx"):
                m = node & (choice[ly, lx] == ci)
                if not m.any():
                    continue
                labels[ci] += m.sum()
                st = kids(state, ly, lx, o)
                code = (st * 3 ** np.arange(st.shape[-1])).sum(-1)[m] + 100 * ci
                for c, k in zip(*np.unique(code, return_counts=True)):
                    hist[c] = hist.get(c, 0) + k
                np.add.at(per_depth, lvl[ly, lx][m], 1)
                ky, kx = (ly + (o != "x"), lx + (o != "y"))
                rep = m
                if o != "x": rep = np.repeat(rep, 2, 1)
                if o != "y": rep = np.repeat(rep, 2, 2)
                reach[ky, kx] |= rep
                d = lvl[ly, lx] * m
                if o != "x": d = np.repeat(d, 2, 1)
                if o != "y": d = np.repeat(d, 2, 2)
                lvl[ky, kx] = np.where(rep, d + 1, lvl[ky, kx])
    p = np.array(list(hist.values()), float); p /= p.sum()
    H = -(p * np.log2(p)).sum()
    tok = cost[0, 0][:, 0, 0]
    return {"tokens": tok.mean(), "p95": np.percentile(tok, 95), "hojas": leaves[0, 0].mean(),
            "prof_media": depth[0, 0].mean(), "prof_max": depth[0, 0].max(), "bits_token": H,
            "bits_imagen": H * tok.mean(), "simbolos": len(hist),
            "cortes_ambos_y_x_%": np.round(100 * labels / labels.sum(), 1).tolist(),
            "tokens_por_nivel": np.round(per_depth / n, 1).tolist()}


if __name__ == "__main__":
    for name, N in (("mnist", 3000), ("quickdraw", 3000), ("quickdraw128", 1000)):
        imgs, _ = load_images(name, train=False)
        imgs = np.ascontiguousarray(imgs[:N]).astype(np.uint8)
        print(f"\n== {name}: {len(imgs)} imágenes de {imgs.shape[1]}x{imgs.shape[2]}, "
              f"{imgs.reshape(len(imgs), -1).sum(1).mean():.0f} píxeles llenos de media")
        for kind in ("quad", "omni", "bin"):
            r = analyse(imgs, kind)
            print(f"{kind:5s} tokens {r['tokens']:7.1f} (p95 {r['p95']:.0f}) | hojas {r['hojas']:7.1f} | "
                  f"prof. {r['prof_media']:.1f} (máx {r['prof_max']}) | {r['bits_token']:.2f} bits/token "
                  f"-> {r['bits_imagen']:.0f} bits/imagen | {r['simbolos']} símbolos | "
                  f"cortes ambos/y/x % {r['cortes_ambos_y_x_%']}")
            print("      tokens por nivel:", r["tokens_por_nivel"])
