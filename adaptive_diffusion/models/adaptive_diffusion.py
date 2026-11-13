"""
Difusión discreta enmascarada sobre un quadtree, de grueso a fino (variante A).

Entrenamiento: se elige un nivel objetivo l; los nodos de niveles < l se dan
limpios como contexto, los de nivel l se enmascaran con probabilidad t ~ U(0,1)
y la red aprende a predecir su estado (entropía cruzada).

Muestreo: se parte de la raíz enmascarada. En cada nivel se desenmascaran los
nodos en varios pasos; los que resultan SPLIT crean 4 hijos enmascarados en el
nivel siguiente. La secuencia nunca contiene nodos que no existen.
"""
import math
import torch
import torch.nn.functional as F

from models.quadtree_transformer import MASK, PAD
from structures.binary_quadtree import SPLIT


def make_training_batch(token_list, depth, generator=None):
    """token_list: lista de tensores (L_i, 4) con columnas (nivel, y, x, estado).
    Devuelve dict con tensores (B, L) rellenados y la máscara de pérdida."""
    seqs, targets = [], []
    for tok in token_list:
        l = int(torch.randint(0, depth + 1, (1,), generator=generator))
        tok = tok[tok[:, 0] <= l]
        if not (tok[:, 0] == l).any():       # el árbol acaba antes: usar su último nivel
            l = int(tok[:, 0].max())
        at_level = tok[:, 0] == l
        t = torch.rand(1, generator=generator).clamp_min(1e-3)
        masked = at_level & (torch.rand(len(tok), generator=generator) < t)
        if not masked.any():                 # al menos un nodo que predecir
            idx = at_level.nonzero()[:, 0]
            masked[idx[torch.randint(0, len(idx), (1,), generator=generator)]] = True
        inp = tok.clone()
        inp[masked, 3] = MASK
        seqs.append(inp)
        targets.append(torch.where(masked, tok[:, 3], torch.full_like(tok[:, 3], -100)))
    n = max(len(s) for s in seqs)
    batch = torch.zeros(len(seqs), n, 4, dtype=torch.long)
    batch[:, :, 3] = PAD
    target = torch.full((len(seqs), n), -100, dtype=torch.long)
    for i, (s, tg) in enumerate(zip(seqs, targets)):
        batch[i, :len(s)] = s
        target[i, :len(s)] = tg
    return {"level": batch[..., 0], "y": batch[..., 1], "x": batch[..., 2], "state": batch[..., 3],
            "pad_mask": batch[..., 3] == PAD, "target": target}


def loss_fn(model, batch, label=None):
    logits = model(batch["state"], batch["level"], batch["y"], batch["x"], batch["pad_mask"], label)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch["target"].reshape(-1), ignore_index=-100)


def _pick_remask(cand, conf, n, mode, generator):
    """Elige n índices de `cand` para volver a enmascarar. 'uniform': al azar;
    'conf': con más probabilidad los que se decidieron con menos confianza."""
    if n <= 0 or len(cand) == 0:
        return cand[:0]
    n = min(n, len(cand))
    if mode == "conf":
        w = (1.0 - conf[cand]).clamp_min(1e-4)
        return cand[torch.multinomial(w, n, replacement=False, generator=generator)]
    return cand[torch.randperm(len(cand), generator=generator)[:n]]


@torch.no_grad()
def sample(model, num, steps_per_level=8, label=None, temperature=1.0, device="cpu", generator=None,
           remask=0.0, remask_rounds=0, remask_mode="conf", stats=None):
    """Genera `num` árboles. Devuelve (lista de tensores (L_i, 4), nº de evaluaciones de red, tokens procesados por muestra).

    Reenmascarado (al estilo ReMDM, sin reentrenar): un nodo ya decidido del nivel en curso
    puede volver a MASK y predecirse otra vez. Sólo se toca el nivel en curso, antes de
    crear sus hijos, porque la red sólo ha visto máscaras en el nivel más profundo presente.
      remask         fracción de los nodos decididos del nivel que se reenmascara cada vez
      remask_rounds  rondas extra de corrección al acabar cada nivel (una evaluación cada una);
                     con 0, el reenmascarado ocurre dentro de los pasos normales del nivel
      remask_mode    'conf' (prioriza los decididos con menos confianza) o 'uniform'
    stats: dict opcional que se rellena con los contadores de reenmascarado."""
    depth = model.depth
    trees = [torch.tensor([[0, 0, 0, MASK]], dtype=torch.long) for _ in range(num)]
    confs = [torch.zeros(1) for _ in range(num)]                    # prob. con la que se decidió cada nodo
    prevs = [torch.full((1,), -1, dtype=torch.long) for _ in range(num)]   # estado antes de reenmascarar
    evals, tokens = 0, 0
    cnt = {"reenmascarados": 0, "cambios": 0, "uniones": 0, "subdivisiones_tardias": 0}

    def predict(active):
        nonlocal evals, tokens
        n = max(len(trees[i]) for i in active)
        batch = torch.zeros(len(active), n, 4, dtype=torch.long)
        batch[:, :, 3] = PAD
        for j, i in enumerate(active):
            batch[j, :len(trees[i])] = trees[i]
        batch = batch.to(device)
        lab = None if label is None else label[active].to(device)
        logits = model(batch[..., 3], batch[..., 0], batch[..., 1], batch[..., 2], batch[..., 3] == PAD, lab)
        evals += 1
        tokens += sum(len(trees[i]) for i in active)
        return F.softmax(logits.float() / temperature, dim=-1).cpu()

    def reveal(i, probs_i, idx):
        if len(idx) == 0:
            return
        t = trees[i]
        new = torch.multinomial(probs_i[idx], 1, generator=generator)[:, 0]
        old = prevs[i][idx]
        was = old >= 0
        cnt["cambios"] += int((was & (new != old)).sum())
        cnt["uniones"] += int((was & (old == SPLIT) & (new != SPLIT)).sum())
        cnt["subdivisiones_tardias"] += int((was & (old != SPLIT) & (new == SPLIT)).sum())
        t[idx, 3] = new
        confs[i][idx] = probs_i[idx].gather(1, new[:, None])[:, 0]
        prevs[i][idx] = -1

    def do_remask(i, l):
        t = trees[i]
        cand = ((t[:, 0] == l) & (t[:, 3] != MASK)).nonzero()[:, 0]
        n = int(torch.bernoulli(torch.full((len(cand),), float(remask)), generator=generator).sum())
        pick = _pick_remask(cand, confs[i], n, remask_mode, generator)
        if len(pick):
            prevs[i][pick] = t[pick, 3]
            t[pick, 3] = MASK
            cnt["reenmascarados"] += len(pick)

    for l in range(depth + 1):
        for step in range(steps_per_level):
            active = [i for i, t in enumerate(trees) if (t[:, 3] == MASK).any()]
            if not active:
                break
            probs = predict(active)
            # calendario coseno: fracción de nodos que siguen enmascarados tras este paso
            keep = math.cos(math.pi / 2 * (step + 1) / steps_per_level)
            last = step == steps_per_level - 1
            for j, i in enumerate(active):
                idx = (trees[i][:, 3] == MASK).nonzero()[:, 0]
                n_keep = int(keep * len(idx)) if not last else 0
                perm = idx[torch.randperm(len(idx), generator=generator)]
                reveal(i, probs[j], perm[: len(idx) - n_keep])
                if remask > 0 and remask_rounds == 0 and not last:
                    do_remask(i, l)
        for _ in range(remask_rounds if remask > 0 else 0):     # corrección al acabar el nivel
            for i in range(num):
                do_remask(i, l)
            active = [i for i, t in enumerate(trees) if (t[:, 3] == MASK).any()]
            if not active:
                break
            probs = predict(active)
            for j, i in enumerate(active):
                reveal(i, probs[j], (trees[i][:, 3] == MASK).nonzero()[:, 0])
        if l < depth:  # los nodos SPLIT de este nivel crean hijos enmascarados
            for i, t in enumerate(trees):
                parents = t[(t[:, 0] == l) & (t[:, 3] == SPLIT)]
                if len(parents) == 0:
                    continue
                off = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
                yx = (2 * parents[:, None, 1:3] + off[None]).reshape(-1, 2)   # orden Morton
                kids = torch.cat([torch.full((len(yx), 1), l + 1), yx, torch.full((len(yx), 1), MASK)], dim=1)
                trees[i] = torch.cat([t, kids], dim=0)
                confs[i] = torch.cat([confs[i], torch.zeros(len(kids))])
                prevs[i] = torch.cat([prevs[i], torch.full((len(kids),), -1, dtype=torch.long)])
    if stats is not None:
        stats.update({k: v / num for k, v in cnt.items()})
    return trees, evals, tokens / num
