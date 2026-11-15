"""
Generación nivel a nivel sobre árboles de cajas 3D (mismo esquema que `tile` en 2D).

Entrenamiento: se elige una profundidad l del árbol; los tokens de profundidad < l se dan limpios y los
de profundidad l se tapan con probabilidad t ~ U(0,1). La red predice las piezas de los tapados.

Muestreo: se parte de la raíz tapada. En cada profundidad se destapan los tokens en varios pasos
(calendario coseno); cada hijo con contenido que no sea hoja llena crea un token tapado en la
profundidad siguiente. Vale igual para octree (`tile`, `oct`) y omnitree (`omni`).
"""
import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from models.box_transformer import CUT_MASK, CUT_PAD, OCC_MASK, OCC_PAD
from structures.boxtree import C_CUT, C_DEPTH, C_FULL, C_IDX, C_LEV, C_OCC, build_tokens

AUGMENTATIONS = 8      # 4 giros de 90° sobre la vertical (eje 1) × espejo


def augment(vox, k):
    """Variante k (0..7) de un lote de rejillas (n, S, S, S): giro de 90°·(k % 4) sobre el eje 1 y espejo si k >= 4."""
    v = np.rot90(vox, k % 4, axes=(1, 3))
    return np.ascontiguousarray(v[:, ::-1] if k >= 4 else v)


def tile_token_count(vox):
    """Nº de tokens del octree `tile` de cada rejilla (n, S, S, S), sin construir el árbol."""
    n, cur, total = len(vox), vox.astype(bool), np.zeros(len(vox), np.int64)
    while cur.shape[1] > 1:
        s = cur.shape[1] // 2
        cur = cur.reshape(n, s, 2, s, 2, s, 2).any((2, 4, 6))
        total += cur.reshape(n, -1).sum(1)
    return total


def _build_job(job):
    vox, k, kind = job
    return build_tokens(augment(vox, k), kind)


class BoxDataset(Dataset):
    """Árboles de un conjunto de rejillas, con sus variantes giradas, como un solo array de tokens."""

    def __init__(self, vox, labels, kind, aug=1, cache=None, chunk=64, log=print, workers=0):
        self.depth = int(vox.shape[1]).bit_length() - 1
        self.kind = kind
        if cache and os.path.exists(cache):
            d = np.load(cache)
            self.tokens, self.offsets, self.labels = d["tokens"], d["offsets"], d["labels"]
            return
        jobs = [(vox[i:i + chunk], k, kind) for k in range(aug) for i in range(0, len(vox), chunk)]
        toks, labs = [], [labels] * aug
        if workers > 1 and len(jobs) > 1:                    # los árboles se calculan en varios procesos
            from multiprocessing import Pool
            with Pool(workers) as pool:
                for done, part in enumerate(pool.imap(_build_job, jobs), 1):
                    toks += part
                    if log and done % 100 == 0:
                        log(f"  árboles '{kind}': {done}/{len(jobs)} bloques")
        else:
            for done, job in enumerate(jobs, 1):
                toks += _build_job(job)
                if log and done % 100 == 0:
                    log(f"  árboles '{kind}': {done}/{len(jobs)} bloques")
        self.offsets = np.concatenate([[0], np.cumsum([len(t) for t in toks])])
        self.tokens = np.concatenate(toks).astype(np.int16)
        self.labels = np.concatenate(labs).astype(np.int64)
        if cache:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            tmp = f"{cache}.{os.getpid()}.tmp.npz"            # escritura atómica: dos procesos pueden coincidir
            np.savez(tmp, tokens=self.tokens, offsets=self.offsets, labels=self.labels)
            os.replace(tmp, cache)

    def __len__(self):
        return len(self.labels)

    def length(self, i):
        return int(self.offsets[i + 1] - self.offsets[i])

    def __getitem__(self, i):
        return torch.from_numpy(self.tokens[self.offsets[i]:self.offsets[i + 1]].astype(np.int64)), int(self.labels[i])


class BucketSampler(Sampler):
    """Lotes de árboles de tamaño parecido (menos relleno), en orden aleatorio."""

    def __init__(self, lengths, batch, pool=50, seed=0):
        self.lengths, self.batch, self.pool = np.asarray(lengths), batch, pool
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.lengths) // self.batch

    def __iter__(self):
        order = self.rng.permutation(len(self.lengths))
        batches = []
        for i in range(0, len(order), self.batch * self.pool):
            part = order[i:i + self.batch * self.pool]
            part = part[np.argsort(self.lengths[part], kind="stable")]
            batches += [part[j:j + self.batch] for j in range(0, len(part) - self.batch + 1, self.batch)]
        for j in self.rng.permutation(len(batches)):
            yield batches[j].tolist()


def collate(seqs, masks=None):
    """Lista de tensores (L_i, 10) -> lote relleno. masks: lista de (L_i,) bool con los tokens tapados."""
    n = max(len(s) for s in seqs)
    tok = torch.zeros(len(seqs), n, 10, dtype=torch.long)
    pad = torch.ones(len(seqs), n, dtype=torch.bool)
    masked = torch.zeros(len(seqs), n, dtype=torch.bool)
    for i, s in enumerate(seqs):
        tok[i, :len(s)] = s
        pad[i, :len(s)] = False
        if masks is not None:
            masked[i, :len(s)] = masks[i]
    cut, occ, full = tok[..., C_CUT].clone(), tok[..., C_OCC].clone(), tok[..., C_FULL].clone()
    cut[masked], occ[masked], full[masked] = CUT_MASK, OCC_MASK, OCC_MASK
    cut[pad], occ[pad], full[pad] = CUT_PAD, OCC_PAD, OCC_PAD
    return {"depth": tok[..., C_DEPTH], "lev": tok[..., C_LEV], "idx": tok[..., C_IDX], "cut": cut, "occ": occ,
            "full": full, "pad": pad, "masked": masked, "target_lev": tok[..., C_LEV],
            "target_cut": tok[..., C_CUT], "target_occ": tok[..., C_OCC], "target_full": tok[..., C_FULL]}


def make_training_batch(items, generator=None):
    """items: lista de (tokens (L, 10), etiqueta). Devuelve (lote, etiquetas)."""
    seqs, masks = [], []
    for tok, _ in items:
        l = int(torch.randint(0, int(tok[:, C_DEPTH].max()) + 1, (1,), generator=generator))
        tok = tok[tok[:, C_DEPTH] <= l]
        at = tok[:, C_DEPTH] == l
        t = torch.rand(1, generator=generator).clamp_min(1e-3)
        m = at & (torch.rand(len(tok), generator=generator) < t)
        if not m.any():
            idx = at.nonzero()[:, 0]
            m[idx[torch.randint(0, len(idx), (1,), generator=generator)]] = True
        seqs.append(tok)
        masks.append(m)
    return collate(seqs, masks), torch.tensor([y for _, y in items])


def _spawn(tok, d, D):
    """Hijos mixtos de los tokens de profundidad d: filas nuevas (tapadas) para la profundidad d + 1."""
    par = tok[tok[:, C_DEPTH] == d]
    new = []
    for cut in np.unique(par[:, C_CUT]):
        p = par[par[:, C_CUT] == cut]
        axes = [a for a in range(3) if (cut >> a) & 1]
        nk = 2 ** len(axes)
        bits = np.array([(cut >> a) & 1 for a in range(3)])
        lev2 = p[:, C_LEV] + bits
        if (lev2 == D).all(1).all():                                 # todos los hijos son vóxeles
            continue
        j = np.arange(nk)
        off = np.zeros((nk, 3), np.int64)
        for q, a in enumerate(axes):
            off[:, a] = (j >> q) & 1
        kid = p[:, None, C_IDX] * (1 + bits) + off[None]             # (P, nk, 3)
        want = ((p[:, None, C_OCC] >> j) & 1).astype(bool) & ~((p[:, None, C_FULL] >> j) & 1).astype(bool)
        want &= ~(lev2 == D).all(1)[:, None]
        who, which = np.nonzero(want)
        rows = np.zeros((len(who), 10), np.int64)
        rows[:, C_DEPTH] = d + 1
        rows[:, C_LEV] = lev2[who]
        rows[:, C_IDX] = kid[who, which]
        rows[:, C_CUT], rows[:, C_OCC], rows[:, C_FULL] = CUT_MASK, OCC_MASK, OCC_MASK
        new.append(rows)
    return np.concatenate(new) if new else np.zeros((0, 10), np.int64)


@torch.no_grad()
def sample(model, num, steps_per_level=8, label=None, temperature=1.0, device="cpu", generator=None,
           max_tokens=None):
    """Genera `num` árboles. Devuelve (lista de arrays (L_i, 10), estadísticas).
    max_tokens: los árboles que lo superan dejan de crecer (quedan incompletos y se cuentan aparte)."""
    D = model.depth
    root = np.zeros((1, 10), np.int64)
    root[0, C_CUT], root[0, C_OCC], root[0, C_FULL] = CUT_MASK, OCC_MASK, OCC_MASK
    trees = [root.copy() for _ in range(num)]
    evals, tokens, truncated = 0, 0, np.zeros(num, bool)
    for d in range(3 * D + 1):
        for step in range(steps_per_level):
            active = [i for i, t in enumerate(trees) if (t[:, C_CUT] == CUT_MASK).any()]
            if not active:
                break
            keep = math.cos(math.pi / 2 * (step + 1) / steps_per_level)
            picks = []
            for i in active:
                idx = np.nonzero(trees[i][:, C_CUT] == CUT_MASK)[0]
                n_keep = int(keep * len(idx)) if step < steps_per_level - 1 else 0
                perm = torch.randperm(len(idx), generator=generator).numpy()
                picks.append(idx[perm[: len(idx) - n_keep]])
            b = collate([torch.from_numpy(trees[i]) for i in active])
            b = {k: v.to(device) for k, v in b.items()}
            lab = None if label is None else label[active].to(device)
            h = model.trunk(b, lab)
            evals += 1
            tokens += sum(len(trees[i]) for i in active)
            rows = torch.cat([torch.full((len(p),), j) for j, p in enumerate(picks)]).to(device)
            cols = torch.from_numpy(np.concatenate(picks)).to(device)
            if len(rows) == 0:
                continue
            cut, occ, full = model.sample_pieces(h[rows, cols].float(), b["lev"][rows, cols], temperature, generator)
            cut, occ, full, k = cut.cpu().numpy(), occ.cpu().numpy(), full.cpu().numpy(), 0
            for i, p in zip(active, picks):
                trees[i][p, C_CUT], trees[i][p, C_OCC], trees[i][p, C_FULL] = cut[k:k + len(p)], occ[k:k + len(p)], full[k:k + len(p)]
                k += len(p)
        grew = False
        for i, t in enumerate(trees):
            if truncated[i]:
                continue
            kids = _spawn(t, d, D)
            if len(kids) == 0:
                continue
            if max_tokens and len(t) + len(kids) > max_tokens:
                truncated[i] = True
                continue
            trees[i] = np.concatenate([t, kids])
            grew = True
        if not grew:
            break
    sizes = np.array([len(t) for t in trees])
    return trees, {"evals": evals, "tokens_por_muestra": tokens / num, "nodos": float(sizes.mean()),
                   "nodos_max": int(sizes.max()), "profundidad": float(np.mean([t[:, C_DEPTH].max() for t in trees])),
                   "truncados": int(truncated.sum())}
