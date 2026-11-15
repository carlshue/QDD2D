"""
Árboles de cajas en 3D como secuencias de tokens: octree y omnitree con el mismo formato.

Una forma es una rejilla binaria S³ (S potencia de 2, D = log2 S). Un token es un nodo mixto del
árbol: una caja diádica con contenido que se parte en hijos. Columnas de un token (COLS):

    depth                  profundidad en el árbol (la raíz es 0); es el "nivel" de generación
    l0, l1, l2             nivel de la caja en cada eje (lado de la caja en el eje a = S / 2^la)
    i0, i1, i2             índice de la caja en cada eje, en unidades de su propio lado
    cut                    qué ejes se parten: máscara de bits, bit a = eje a (1..7)
    occ                    qué hijos tienen contenido: máscara de bits sobre los hijos
    full                   cuáles de esos hijos son cajas llenas (hojas); el resto son mixtos y tienen
                           su propio token un nivel de profundidad más abajo

Hijos de un nodo: si se parten k ejes hay 2^k hijos. El hijo j toma, en el p-ésimo eje partido
(ejes en orden creciente), la mitad dada por el bit p de j.

Tipos de árbol (`kind`):
    "tile"  octree de la bibliografía: se parten siempre los tres ejes y sólo hay hojas llenas en el
            nivel de vóxel (un byte de ocupación por nodo; `full` sólo se usa en el último nivel)
    "oct"   octree con hojas llenas a cualquier nivel
    "omni"  omnitree: cada nodo elige qué ejes partir; es el árbol exacto con menos nodos mixtos,
            calculado por programación dinámica sobre todas las cajas diádicas

Todos son sin pérdida: `tokens_to_voxels(build_tokens(v, kind)) == v`.
"""
import itertools
import numpy as np

COLS = ("depth", "l0", "l1", "l2", "i0", "i1", "i2", "cut", "occ", "full")
C_DEPTH, C_LEV, C_IDX, C_CUT, C_OCC, C_FULL = 0, slice(1, 4), slice(4, 7), 7, 8, 9
KINDS = ("tile", "oct", "omni")
ALL_AXES = 7


def cut_axes(cut):
    """Ejes partidos por una máscara de corte, en orden creciente."""
    return [a for a in range(3) if (cut >> a) & 1]


def child_boxes(lev, idx, cut):
    """Cajas hijas de un nodo. Devuelve (niveles (3,), índices (2^k, 3)) con el orden de hijos del módulo."""
    axes = cut_axes(cut)
    lev2 = np.array(lev) + np.array([(cut >> a) & 1 for a in range(3)])
    out = np.tile(np.array(idx), (2 ** len(axes), 1))
    for p, a in enumerate(axes):
        out[:, a] = 2 * out[:, a] + ((np.arange(len(out)) >> p) & 1)
    return lev2, out


def _cut_options(kind):
    masks = [m for m in itertools.product((0, 1), repeat=3) if any(m)]
    if kind in ("oct", "tile"):
        masks = [m for m in masks if all(m)]
    return sorted(masks, key=lambda m: (-sum(m), m))      # desempate: partir más ejes (árbol menos profundo)


def _mask_id(m):
    return sum(b << a for a, b in enumerate(m))


def build_tokens(vox, kind="omni"):
    """vox: (n, S, S, S) en {0,1}. Devuelve una lista de n arrays (L_i, 10) int16, ordenados por profundidad."""
    assert kind in KINDS
    vox = np.asarray(vox).astype(np.uint8)
    n, S = vox.shape[0], vox.shape[1]
    D = int(np.log2(S))
    assert vox.shape == (n, S, S, S) and 2 ** D == S
    if kind == "tile":
        return _build_tile(vox, D)
    levels = list(itertools.product(range(D + 1), repeat=3))
    add = lambda lv, m: tuple(a + b for a, b in zip(lv, m))
    shape = lambda lv: (n,) + tuple(2 ** l for l in lv)

    def kids(arr, lv, m):
        """Valores de los hijos de cada nodo de nivel lv con el corte m: (n, *shape(lv), 2^|m|)."""
        a = arr[add(lv, m)]
        split = []
        for l, b in zip(lv, m):
            split += [2 ** l, 2 ** b]
        a = a.reshape((n,) + tuple(split))
        return a.transpose(0, 1, 3, 5, 6, 4, 2).reshape(shape(lv) + (-1,))   # primer eje partido = bit menos significativo

    mx, mn = {(D,) * 3: vox}, {(D,) * 3: vox}
    for lv in sorted(levels, key=lambda t: -sum(t)):
        if lv in mx:
            continue
        j = next(i for i in range(3) if lv[i] < D)
        m = tuple(int(i == j) for i in range(3))
        mx[lv], mn[lv] = kids(mx, lv, m).max(-1), kids(mn, lv, m).min(-1)
    uni = {lv: mx[lv] == mn[lv] for lv in levels}
    state = {lv: np.where(uni[lv], mx[lv], 2).astype(np.uint8) for lv in levels}     # 0 vacío, 1 lleno, 2 mixto
    del mn

    all_opts = _cut_options(kind)
    cost, choice = {}, {}
    for lv in sorted(levels, key=lambda t: -sum(t)):
        opts = [m for m in all_opts if all(l + b <= D for l, b in zip(lv, m))]
        u = uni[lv]
        if not opts:
            cost[lv] = np.where(u, 0, 10 ** 7).astype(np.int32)
            choice[lv] = np.zeros(u.shape, np.uint8)
            continue
        cs = np.stack([kids(cost, lv, m).sum(-1) for m in opts])
        best = cs.argmin(0)
        cost[lv] = np.where(u, 0, 1 + np.take_along_axis(cs, best[None], 0)[0]).astype(np.int32)
        choice[lv] = np.where(u, 0, np.array([_mask_id(m) for m in opts], np.uint8)[best])

    root = (0,) * 3
    reach = {lv: np.zeros(shape(lv), bool) for lv in levels}
    reach[root][:] = True
    depth = {lv: np.zeros(shape(lv), np.int16) for lv in levels}
    rows = []
    for lv in sorted(levels, key=sum):
        node = reach[lv] & ~uni[lv]
        if not node.any():
            continue
        for m in all_opts:
            cid = _mask_id(m)
            sel = node & (choice[lv] == cid)
            if not sel.any():
                continue
            who = np.nonzero(sel)                                   # (forma, i0, i1, i2)
            st = kids(state, lv, m)[sel].astype(np.int64)             # (nodos, 2^k)
            bit = 1 << np.arange(st.shape[1])
            occ, full = ((st != 0) * bit).sum(1), ((st == 1) * bit).sum(1)
            block = np.zeros((len(who[0]), 1 + len(COLS)), np.int64)
            block[:, 0] = who[0]
            block[:, 1 + C_DEPTH] = depth[lv][sel]
            block[:, 2:5] = lv
            block[:, 5:8] = np.stack(who[1:], 1)
            block[:, 1 + C_CUT], block[:, 1 + C_OCC], block[:, 1 + C_FULL] = cid, occ, full
            rows.append(block)
            rep, dep = sel, depth[lv] * sel
            for ax, b in enumerate(m):
                if b:
                    rep, dep = np.repeat(rep, 2, ax + 1), np.repeat(dep, 2, ax + 1)
            child = add(lv, m)
            reach[child] |= rep
            depth[child] = np.where(rep, dep + 1, depth[child])
    return _split_rows(rows, n)


def _build_tile(vox, D):
    n = vox.shape[0]
    occ = [vox.astype(bool)]
    for _ in range(D):
        o = occ[-1]
        s = o.shape[1] // 2
        occ.append(o.reshape(n, s, 2, s, 2, s, 2).any((2, 4, 6)))
    occ = occ[::-1]                                              # occ[l]: (n, 2^l, 2^l, 2^l)
    rows = []
    for l in range(D):
        s = 2 ** l
        c = occ[l + 1].reshape(n, s, 2, s, 2, s, 2).transpose(0, 1, 3, 5, 6, 4, 2).reshape(n, s, s, s, 8)
        code = (c * (1 << np.arange(8))).sum(-1)
        keep = occ[l].copy()
        if l == 0:
            keep[:] = True                                       # la raíz existe siempre
        who = np.nonzero(keep)
        block = np.zeros((len(who[0]), 1 + len(COLS)), np.int64)
        block[:, 0] = who[0]
        block[:, 1 + C_DEPTH] = l
        block[:, 2:5] = l
        block[:, 5:8] = np.stack(who[1:], 1)
        block[:, 1 + C_CUT] = ALL_AXES
        block[:, 1 + C_OCC] = code[who]
        block[:, 1 + C_FULL] = code[who] if l == D - 1 else 0     # en el último nivel los hijos son vóxeles
        rows.append(block)
    return _split_rows(rows, n)


def _split_rows(rows, n):
    rows = np.concatenate(rows) if rows else np.zeros((0, 1 + len(COLS)), np.int64)
    order = np.lexsort((rows[:, 1 + C_DEPTH], rows[:, 0]))       # por forma y, dentro, por profundidad (estable)
    rows = rows[order]
    cuts = np.cumsum(np.bincount(rows[:, 0], minlength=n))[:-1]
    return [t.astype(np.int16) for t in np.split(rows[:, 1:], cuts)]


def tokens_to_voxels(tok, D):
    """Inversa de build_tokens para una forma: tok (L, 10) -> rejilla (S, S, S) uint8."""
    S = 2 ** D
    vox = np.zeros((S, S, S), np.uint8)
    tok = np.asarray(tok).astype(np.int64)
    for row in tok:
        full = int(row[C_FULL])
        if not full:
            continue
        lev2, kid = child_boxes(row[C_LEV], row[C_IDX], int(row[C_CUT]))
        size = S >> lev2
        for j in range(len(kid)):
            if (full >> j) & 1:
                lo = kid[j] * size
                vox[lo[0]:lo[0] + size[0], lo[1]:lo[1] + size[1], lo[2]:lo[2] + size[2]] = 1
    return vox


def check_tree(tok, D):
    """Comprueba que una secuencia es un árbol bien formado. Devuelve una lista de errores (vacía si vale)."""
    tok = np.asarray(tok).astype(np.int64)
    errors, expected = [], {(0, (0, 0, 0), (0, 0, 0))}
    seen = set()
    for row in tok:
        key = (int(row[C_DEPTH]), tuple(row[C_LEV]), tuple(row[C_IDX]))
        cut, occ, full = int(row[C_CUT]), int(row[C_OCC]), int(row[C_FULL])
        if key not in expected:
            errors.append(f"nodo sin padre: {key}")
        if key in seen:
            errors.append(f"nodo repetido: {key}")
        seen.add(key)
        nk = 2 ** len(cut_axes(cut))
        if not 1 <= cut <= 7 or any(row[C_LEV][a] >= D for a in cut_axes(cut)):
            errors.append(f"corte inválido {cut} en {key}")
            continue
        if not 1 <= occ < 2 ** nk or full & ~occ:
            errors.append(f"ocupación/lleno inválidos ({occ}, {full}) en {key}")
        lev2, kid = child_boxes(row[C_LEV], row[C_IDX], cut)
        unit = bool((lev2 == D).all())
        for j in range(nk):
            if (occ >> j) & 1 and not (full >> j) & 1:
                if unit:
                    errors.append(f"hijo vóxel mixto en {key}")
                expected.add((key[0] + 1, tuple(lev2), tuple(kid[j])))
    missing = {k for k in expected if k not in seen}
    if missing:
        errors.append(f"faltan {len(missing)} nodos mixtos")
    return errors
