"""
Quadtree binario exacto, vectorizado por lotes (numpy).

Una imagen binaria de lado 2^D se representa por niveles l = 0..D. El nivel l
es una rejilla de 2^l x 2^l celdas, cada una con un estado:

    EMPTY  (0)  bloque completamente vacío  -> hoja
    FULL   (1)  bloque completamente lleno  -> hoja
    SPLIT  (2)  bloque mixto                -> tiene 4 hijos en el nivel l+1
    ABSENT (3)  la celda no existe (su padre no es SPLIT)

En el nivel D (píxeles) sólo hay EMPTY/FULL. La representación es sin pérdida.
"""
import numpy as np

EMPTY, FULL, SPLIT, ABSENT = 0, 1, 2, 3
NUM_STATES = 3  # clases que el modelo predice


def build_levels(imgs):
    """imgs: (N, S, S) con valores {0,1}, S potencia de 2.
    Devuelve lista de D+1 arrays uint8; levels[l] tiene forma (N, 2^l, 2^l)."""
    imgs = np.asarray(imgs).astype(np.uint8)
    n, s, s2 = imgs.shape
    depth = int(np.log2(s))
    assert s == s2 and 2 ** depth == s, "la imagen debe ser cuadrada y potencia de 2"
    levels = [None] * (depth + 1)
    levels[depth] = imgs
    for l in range(depth - 1, -1, -1):  # de fino a grueso: fusionar bloques uniformes
        side = 2 ** l
        c = levels[l + 1].reshape(n, side, 2, side, 2).transpose(0, 1, 3, 2, 4).reshape(n, side, side, 4)
        first = c[..., 0]
        uniform = (c == first[..., None]).all(-1) & (first != SPLIT)
        levels[l] = np.where(uniform, first, SPLIT).astype(np.uint8)
    for l in range(depth):  # de grueso a fino: marcar celdas inexistentes
        parent_split = levels[l] == SPLIT
        exists = np.repeat(np.repeat(parent_split, 2, axis=1), 2, axis=2)
        levels[l + 1] = np.where(exists, levels[l + 1], ABSENT).astype(np.uint8)
    return levels


def levels_to_image(levels):
    """Inversa de build_levels. Devuelve (N, S, S) uint8."""
    depth = len(levels) - 1
    s = 2 ** depth
    img = np.zeros((levels[0].shape[0], s, s), dtype=np.uint8)
    for l, st in enumerate(levels):
        k = s // (2 ** l)
        up = np.repeat(np.repeat(st, k, axis=1), k, axis=2)
        img[up == FULL] = 1
    return img


def _morton(y, x, bits):
    code = np.zeros_like(y, dtype=np.int64)
    for b in range(bits):
        code |= ((x >> b) & 1) << (2 * b)
        code |= ((y >> b) & 1) << (2 * b + 1)
    return code


def level_cells(level_state):
    """Celdas existentes de un nivel (2^l, 2^l) en orden Morton.
    Devuelve (ys, xs, estados)."""
    ys, xs = np.nonzero(level_state != ABSENT)
    bits = max(1, int(np.log2(level_state.shape[0])))
    order = np.argsort(_morton(ys, xs, bits), kind="stable")
    ys, xs = ys[order], xs[order]
    return ys, xs, level_state[ys, xs]


def to_tokens(levels_one):
    """Serializa UN árbol (lista de arrays (2^l, 2^l)) a un array (L, 4) con
    columnas (nivel, y, x, estado), en orden BFS y Morton dentro de cada nivel.
    y, x son coordenadas de celda dentro de su nivel."""
    rows = []
    for l, st in enumerate(levels_one):
        ys, xs, s = level_cells(st)
        rows.append(np.stack([np.full_like(ys, l), ys, xs, s.astype(np.int64)], axis=1))
    return np.concatenate(rows, axis=0)


def tokens_to_levels(tokens, depth):
    """Inversa de to_tokens."""
    levels = [np.full((2 ** l, 2 ** l), ABSENT, dtype=np.uint8) for l in range(depth + 1)]
    for l, y, x, s in tokens:
        levels[l][y, x] = s
    return levels


def node_counts(levels):
    """Nº de nodos existentes por imagen y nivel: array (N, D+1)."""
    return np.stack([(st != ABSENT).reshape(st.shape[0], -1).sum(1) for st in levels], axis=1)
