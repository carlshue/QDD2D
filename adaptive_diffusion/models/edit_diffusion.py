"""
Difusión por edición del árbol.

Se parte de un árbol aleatorio (ruido) y, en cada paso, cualquier nodo de
cualquier nivel puede cambiar de estado:
    hoja -> SPLIT         = subdividir (aparecen 4 hijos)
    SPLIT -> EMPTY/FULL   = juntar (desaparece su subárbol)
    EMPTY <-> FULL        = cambiar el relleno
No hace falta un estado "juntar": lo decide el padre al volverse hoja.

Entrenamiento: se corrompe el árbol real con ruido de nivel t y la red predice
el estado verdadero de cada nodo presente.
Muestreo: al pasar de t a s < t, cada nodo salta a la predicción de la red con
probabilidad (t - s) / t; si no, se queda como está.
"""
import torch
import torch.nn.functional as F

from models.quadtree_transformer import PAD
from structures.binary_quadtree import SPLIT


def _gather(pyr, state, order, pad):
    return {"state": state.gather(1, order).long().masked_fill(pad, PAD),
            "level": pyr.level[order], "y": pyr.y[order], "x": pyr.x[order], "pad_mask": pad}


def training_batch(pyr, s0, prior, generator=None):
    t = torch.rand(s0.shape[0], device=s0.device, generator=generator)
    state, ex = pyr.corrupt(s0, t, prior, generator)
    order, pad = pyr.to_sequence(ex)
    batch = _gather(pyr, state, order, pad)
    batch["target"] = s0.gather(1, order).long().masked_fill(pad, -100)
    batch["t"] = t
    return batch


def loss_fn(model, batch, label=None):
    logits = model(batch["state"], batch["level"], batch["y"], batch["x"], batch["pad_mask"], label, batch["t"])
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), batch["target"].reshape(-1), ignore_index=-100)
    with torch.no_grad():  # acierto sobre los nodos que el ruido había cambiado
        wrong_in = (batch["state"] != batch["target"]) & ~batch["pad_mask"]
        fixed = (logits.argmax(-1) == batch["target"]) & wrong_in
        acc = fixed.sum().float() / wrong_in.sum().clamp_min(1)
    return loss, acc


@torch.no_grad()
def sample(model, pyr, prior, num, steps=32, label=None, temperature=1.0, t_complete=0.5, generator=None):
    """Devuelve (imágenes bool (num, S, S), estadísticas)."""
    dev = prior.device
    state = pyr.sample_states(prior, num, generator)
    ex = pyr.exists(state)
    ts = torch.linspace(1.0, 0.0, steps + 1).tolist()
    evals, tokens, start_nodes = 0, 0, ex.sum(1).float().mean().item()

    def predict(t):
        nonlocal evals, tokens
        order, pad = pyr.to_sequence(ex)
        b = _gather(pyr, state, order, pad)
        tt = torch.full((num,), t, device=dev)
        logits = model(b["state"], b["level"], b["y"], b["x"], pad, label, tt).float()
        evals += 1
        tokens += int((~pad).sum())
        pick = torch.multinomial(F.softmax(logits / temperature, -1).reshape(-1, logits.shape[-1]), 1,
                                 generator=generator).reshape(num, -1)
        x0 = torch.zeros_like(state)
        x0.scatter_(1, order, pick.to(state.dtype).masked_fill(pad, 0))
        return x0

    def refresh(old_ex):
        """Recalcula qué celdas existen; las recién creadas arrancan como ruido."""
        new_ex = pyr.exists(state)
        fresh = new_ex & ~old_ex
        state[fresh] = pyr.sample_states(prior, num, generator)[fresh]
        # un hijo nuevo que sale SPLIT crea a su vez hijos: repetir hasta estabilizar
        while True:
            nxt = pyr.exists(state)
            more = nxt & ~new_ex
            if not more.any():
                return nxt, (nxt & ~old_ex)
            state[more] = pyr.sample_states(prior, num, generator)[more]
            new_ex = nxt

    fresh = torch.zeros_like(ex)
    for t, s in zip(ts[:-1], ts[1:]):
        x0 = predict(t)
        jump = ex & (torch.rand(state.shape, device=dev, generator=generator) < (t - s) / t)
        state = torch.where(jump, x0, state)
        ex, fresh = refresh(ex)
    # cierre: los nodos creados en el último paso aún son ruido; se resuelven aparte
    for _ in range(pyr.depth + 1):
        if not fresh.any():
            break
        x0 = predict(t_complete)
        state = torch.where(fresh, x0, state)
        ex, fresh = refresh(ex)
    stats = {"evals": evals, "tokens_por_muestra": tokens / num, "nodos_inicio": start_nodes,
             "nodos_final": ex.sum(1).float().mean().item(), "nodos_max": int(ex.sum(1).max()),
             "sin_resolver": int(fresh.sum())}
    return pyr.to_image(state, ex), stats
