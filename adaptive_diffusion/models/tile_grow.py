"""
Crecimiento por patrones de ocupación ('tile'): un token por nodo subdividido.

En vez de un token por nodo con estado {EMPTY, FULL, SPLIT}, sólo tienen token los nodos
que contienen algo, y su valor es qué hijos contienen algo: 4 bits, uno por hijo en orden
Morton (bit k = hijo dy=k//2, dx=k%2). En el último nivel con token (depth-1) los hijos
son píxeles. Los hijos vacíos no aparecen en la secuencia. Es sin pérdida: un bloque
lleno se describe bajando hasta los píxeles.

Token: (nivel, y, x, código), nivel en 0..depth-1, código en 0..15 (0 sólo en una
imagen vacía). Ids de entrada extra: T_MASK=16, T_PAD=17.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

T_STATES = 16
T_MASK = T_STATES
T_PAD = T_STATES + 1
_OFF = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])


def images_to_tokens(imgs):
    """imgs (N, S, S) en {0,1}. Devuelve lista de N tensores (L_i, 4) long, ordenados por nivel."""
    x = torch.as_tensor(imgs).bool()
    n, s = x.shape[0], x.shape[1]
    depth = s.bit_length() - 1
    occ = [x]                                           # occ[0] = píxeles ... hasta la raíz
    for _ in range(depth):
        o = occ[-1]
        occ.append(o.reshape(n, o.shape[1] // 2, 2, o.shape[2] // 2, 2).any(4).any(2))
    occ = occ[::-1]                                     # occ[l]: (N, 2^l, 2^l)
    rows = []
    for l in range(depth):
        c = occ[l + 1].reshape(n, 2 ** l, 2, 2 ** l, 2).permute(0, 1, 3, 2, 4).reshape(n, 2 ** l, 2 ** l, 4).long()
        code = (c * torch.tensor([1, 2, 4, 8])).sum(-1)
        keep = occ[l].clone()
        if l == 0:
            keep[:] = True                              # la raíz existe siempre
        i, y, xx = keep.nonzero(as_tuple=True)
        rows.append(torch.stack([i, torch.full_like(i, l), y, xx, code[i, y, xx]], 1))
    rows = torch.cat(rows)
    rows = rows[torch.sort(rows[:, 0], stable=True).indices]
    return list(torch.split(rows[:, 1:].contiguous(), torch.bincount(rows[:, 0], minlength=n).tolist()))


def tokens_to_image(tok, depth):
    """tok (L, 4) sin máscaras -> imagen (S, S) uint8."""
    s = 2 ** depth
    img = torch.zeros(s, s, dtype=torch.uint8)
    t = tok[tok[:, 0] == depth - 1]
    for k in range(4):
        on = ((t[:, 3] >> k) & 1).bool() & (t[:, 3] < T_STATES)
        img[2 * t[on, 1] + k // 2, 2 * t[on, 2] + k % 2] = 1
    return img


class TileDataset(Dataset):
    def __init__(self, train=True, limit=None, dataset="quickdraw"):
        from data.datasets import load_images
        imgs, labels = load_images(dataset, train)
        if limit:
            imgs, labels = imgs[:limit], labels[:limit]
        self.depth = int(imgs.shape[1]).bit_length() - 1
        self.tokens = images_to_tokens(imgs)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        return self.tokens[i], self.labels[i]


def make_training_batch(token_list, depth, generator=None):
    """Igual que en 'grow': niveles < l limpios, nivel l tapado con probabilidad t."""
    seqs, targets = [], []
    for tok in token_list:
        l = int(torch.randint(0, depth, (1,), generator=generator))
        tok = tok[tok[:, 0] <= l]
        l = int(tok[:, 0].max())
        at_level = tok[:, 0] == l
        t = torch.rand(1, generator=generator).clamp_min(1e-3)
        masked = at_level & (torch.rand(len(tok), generator=generator) < t)
        if not masked.any():
            idx = at_level.nonzero()[:, 0]
            masked[idx[torch.randint(0, len(idx), (1,), generator=generator)]] = True
        inp = tok.clone()
        inp[masked, 3] = T_MASK
        seqs.append(inp)
        targets.append(torch.where(masked, tok[:, 3], torch.full_like(tok[:, 3], -100)))
    n = max(len(s) for s in seqs)
    batch = torch.zeros(len(seqs), n, 4, dtype=torch.long)
    batch[:, :, 3] = T_PAD
    target = torch.full((len(seqs), n), -100, dtype=torch.long)
    for i, (s, tg) in enumerate(zip(seqs, targets)):
        batch[i, :len(s)] = s
        target[i, :len(s)] = tg
    return {"level": batch[..., 0], "y": batch[..., 1], "x": batch[..., 2], "state": batch[..., 3],
            "pad_mask": batch[..., 3] == T_PAD, "target": target}


@torch.no_grad()
def sample(model, num, steps_per_level=8, label=None, temperature=1.0, device="cpu", generator=None):
    """Devuelve (lista de tensores (L_i, 4), evaluaciones de red, tokens procesados por muestra)."""
    depth = model.depth
    trees = [torch.tensor([[0, 0, 0, T_MASK]], dtype=torch.long) for _ in range(num)]
    evals, tokens = 0, 0
    for l in range(depth):
        for step in range(steps_per_level):
            active = [i for i, t in enumerate(trees) if (t[:, 3] == T_MASK).any()]
            if not active:
                break
            n = max(len(trees[i]) for i in active)
            batch = torch.zeros(len(active), n, 4, dtype=torch.long)
            batch[:, :, 3] = T_PAD
            for j, i in enumerate(active):
                batch[j, :len(trees[i])] = trees[i]
            batch = batch.to(device)
            lab = None if label is None else label[active].to(device)
            logits = model(batch[..., 3], batch[..., 0], batch[..., 1], batch[..., 2], batch[..., 3] == T_PAD, lab)
            evals += 1
            tokens += sum(len(trees[i]) for i in active)
            probs = F.softmax(logits.float() / temperature, dim=-1).cpu()
            keep = math.cos(math.pi / 2 * (step + 1) / steps_per_level)
            last = step == steps_per_level - 1
            for j, i in enumerate(active):
                idx = (trees[i][:, 3] == T_MASK).nonzero()[:, 0]
                n_keep = int(keep * len(idx)) if not last else 0
                pick = idx[torch.randperm(len(idx), generator=generator)][: len(idx) - n_keep]
                if len(pick):
                    trees[i][pick, 3] = torch.multinomial(probs[j][pick], 1, generator=generator)[:, 0]
        if l < depth - 1:      # cada bit a 1 crea un hijo tapado
            for i, t in enumerate(trees):
                par = t[t[:, 0] == l]
                bits = ((par[:, 3:4] >> torch.arange(4)) & 1).bool()          # (P, 4)
                yx = (2 * par[:, None, 1:3] + _OFF[None])[bits]
                if len(yx):
                    kids = torch.cat([torch.full((len(yx), 1), l + 1), yx, torch.full((len(yx), 1), T_MASK)], 1)
                    trees[i] = torch.cat([t, kids], 0)
    return trees, evals, tokens / num
