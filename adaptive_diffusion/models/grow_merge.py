"""
Crecimiento con unión: el árbol crece de la raíz a los píxeles como en `grow`,
pero los nodos ya decididos se pueden revisar.

No hay árbol aleatorio de partida: los nodos nacen sin decidir (MASK). En cada
paso la red predice el estado de TODOS los nodos presentes:
    nodo MASK                -> se decide (según el calendario del nivel)
    hoja decidida -> SPLIT   = subdividir más tarde (nacen 4 hijos MASK)
    SPLIT decidido -> hoja   = juntar (desaparece su subárbol)
    EMPTY <-> FULL           = corregir el relleno

Entrenamiento: se elige un nivel frontera l. Los nodos del nivel l se tapan con
prob. t; los de niveles anteriores se estropean un poco (con prob. eps se
sustituyen por un estado al azar y con prob. eps/2 se tapan), que es lo que la
red se encontrará al muestrear tras sus propios errores. La red predice el
estado verdadero de cada nodo presente.
"""
import math
import torch
import torch.nn.functional as F

from models.edit_diffusion import _gather, predict_states
from models.quadtree_transformer import MASK
from structures.binary_quadtree import SPLIT


def training_batch(pyr, s0, prior, eps_max=0.15, generator=None):
    n, dev = s0.shape[0], s0.device
    lvl = torch.randint(0, pyr.depth + 1, (n, 1), device=dev, generator=generator)
    t = torch.rand(n, device=dev, generator=generator)
    eps = eps_max * torch.rand(n, 1, device=dev, generator=generator)
    below, at = pyr.level[None] < lvl, pyr.level[None] == lvl
    u = torch.rand(s0.shape, device=dev, generator=generator)
    state = torch.where(below & (u < eps), pyr.sample_states(prior, n, generator), s0)
    u = torch.rand(s0.shape, device=dev, generator=generator)
    hide = (at & (u < t[:, None])) | (below & (u < eps / 2))
    state = torch.where(hide, torch.full_like(state, MASK), state)
    ex = pyr.exists(state) & (pyr.level[None] <= lvl)      # un nodo MASK no tiene hijos
    order, pad = pyr.to_sequence(ex)
    batch = _gather(pyr, state, order, pad)
    batch["target"] = s0.gather(1, order).long().masked_fill(pad, -100)
    batch["t"] = t
    return batch


def loss_fn(model, batch, label=None):
    logits = model(batch["state"], batch["level"], batch["y"], batch["x"], batch["pad_mask"], label, batch["t"])
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), batch["target"].reshape(-1), ignore_index=-100)
    with torch.no_grad():  # acierto sobre los nodos decididos que estaban mal
        wrong_in = (batch["state"] != batch["target"]) & (batch["state"] != MASK) & ~batch["pad_mask"]
        fixed = (logits.argmax(-1) == batch["target"]) & wrong_in
        acc = fixed.sum().float() / wrong_in.sum().clamp_min(1)
    return loss, acc


@torch.no_grad()
def sample(model, pyr, num, steps_per_level=8, label=None, temperature=1.0, revise=1.0, revise_thr=0.5,
           generator=None, trace=None):
    """Devuelve (imágenes bool (num, S, S), estadísticas).

    revise: prob. con la que, en cada paso, se revisa un nodo ya decidido cuyo
        estado actual recibe de la red una probabilidad menor que `revise_thr`.
        revise=0 desactiva la unión (crecimiento puro). revise_thr=1 revisa
        cualquier nodo, no sólo los dudosos."""
    dev = pyr.level.device
    lev = pyr.level[None]
    state = torch.full((num, pyr.size), MASK, dtype=torch.uint8, device=dev)
    ex = torch.zeros(num, pyr.size, dtype=torch.bool, device=dev)
    ex[:, 0] = True
    evals, tokens, changes, merges, splits = 0, 0, 0, 0, 0

    def refresh(old_ex, l):
        """Qué nodos existen hasta el nivel l; los que nacen arrancan sin decidir."""
        fresh = pyr.exists(state) & (lev <= l) & ~old_ex
        state[fresh] = MASK
        return pyr.exists(state) & (lev <= l)

    def step(l, t, n_keep_frac, do_revise, tag):
        nonlocal state, ex, evals, tokens, changes, merges, splits
        x0, p_cur, n_tok = predict_states(model, pyr, state, ex, t, label, temperature, generator)
        evals += 1
        tokens += n_tok
        masked = ex & (state == MASK)
        # destapar: se quedan tapados floor(n_keep_frac * n) nodos de cada árbol, al azar
        score = torch.rand(state.shape, device=dev, generator=generator).masked_fill(~masked, 2.0)
        rank = score.argsort(1).argsort(1)
        n_masked = masked.sum(1, keepdim=True)
        reveal = masked & (rank < n_masked - (n_keep_frac * n_masked).floor().long())
        new = torch.where(reveal, x0, state)
        if do_revise and revise > 0:
            rev = ex & ~masked & (p_cur < revise_thr) & (x0 != state)
            if revise < 1:
                rev &= torch.rand(state.shape, device=dev, generator=generator) < revise
            changes += int(rev.sum())
            merges += int((rev & (state == SPLIT)).sum())
            splits += int((rev & (x0 == SPLIT)).sum())
            new = torch.where(rev, x0, new)
        if trace is not None:
            trace.record(ex, p_cur, new != state, tag)
        state = new
        ex = refresh(ex, l)

    for l in range(pyr.depth + 1):
        ex = refresh(ex, l)
        for k in range(steps_per_level):
            if not (ex & (state == MASK)).any():
                break
            keep = math.cos(math.pi / 2 * (k + 1) / steps_per_level) if k < steps_per_level - 1 else 0.0
            step(l, math.cos(math.pi / 2 * k / steps_per_level), keep, True, f"nivel{l}")
    # cierre: nodos nacidos de una revisión en el último paso
    for _ in range(pyr.depth + 1):
        if not (ex & (state == MASK)).any():
            break
        step(pyr.depth, 0.0, 0.0, False, "cierre")
    stats = {"evals": evals, "tokens_por_muestra": tokens / num, "nodos_final": ex.sum(1).float().mean().item(),
             "nodos_max": int(ex.sum(1).max()), "revisiones": changes / num, "uniones": merges / num,
             "subdivisiones_tardias": splits / num, "sin_resolver": int((ex & (state == MASK)).sum())}
    return pyr.to_image(state, ex), stats
