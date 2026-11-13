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
def predict_states(model, pyr, state, ex, t, label=None, temperature=1.0, generator=None):
    """Evalúa la red sobre los nodos presentes. Devuelve, en forma de pirámide
    (num, size): un estado sorteado de la predicción, la probabilidad que la red
    da al estado actual de cada nodo (0 si está sin decidir) y el nº de tokens."""
    num = state.shape[0]
    order, pad = pyr.to_sequence(ex)
    b = _gather(pyr, state, order, pad)
    tt = torch.full((num,), float(t), device=state.device)
    logits = model(b["state"], b["level"], b["y"], b["x"], pad, label, tt).float()
    probs = F.softmax(logits / temperature, -1)
    pick = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1, generator=generator).reshape(num, -1)
    cur = b["state"].clamp(max=probs.shape[-1] - 1)
    p_cur = probs.gather(-1, cur[..., None])[..., 0].masked_fill(b["state"] >= probs.shape[-1], 0.0)
    x0 = torch.zeros_like(state)
    x0.scatter_(1, order, pick.to(state.dtype).masked_fill(pad, 0))
    p_current = torch.zeros(state.shape, device=state.device)
    p_current.scatter_(1, order, p_cur.masked_fill(pad, 0.0))
    return x0, p_current, int((~pad).sum())


@torch.no_grad()
def sample(model, pyr, prior, num, steps=32, label=None, temperature=1.0, t_complete=0.5, generator=None,
           resolve_fresh=False, fresh_prior=None, init_state=None, trace=None):
    """Devuelve (imágenes bool (num, S, S), estadísticas).

    resolve_fresh: los nodos que nacen en un paso (hijos de un SPLIT nuevo) se
        resuelven en ese mismo paso con la red, en vez de quedarse como ruido
        puro hasta que les toque saltar. Corrige el desajuste con el
        entrenamiento (allí, a nivel de ruido t, un hijo es ruido sólo con
        prob. t) a cambio de más evaluaciones de red por paso.
    fresh_prior: ruido con el que nacen los nodos nuevos (por defecto, `prior`).
    init_state: árbol de partida (num, size); por defecto se sortea de `prior`.
    Los dos últimos sirven para estudiar cuánto pesa el árbol aleatorio.
    trace: un models.heat.HeatTrace para anotar nodos calientes (no altera nada)."""
    dev = prior.device
    fresh_prior = prior if fresh_prior is None else fresh_prior
    state = pyr.sample_states(prior, num, generator) if init_state is None else init_state.clone()
    ex = pyr.exists(state)
    ts = torch.linspace(1.0, 0.0, steps + 1).tolist()
    evals, tokens, start_nodes = 0, 0, ex.sum(1).float().mean().item()

    def predict(t):
        nonlocal evals, tokens
        x0, p_cur, n_tok = predict_states(model, pyr, state, ex, t, label, temperature, generator)
        evals += 1
        tokens += n_tok
        return x0, p_cur

    def refresh(old_ex):
        """Recalcula qué celdas existen; las recién creadas arrancan como ruido."""
        new_ex = pyr.exists(state)
        fresh = new_ex & ~old_ex
        state[fresh] = pyr.sample_states(fresh_prior, num, generator)[fresh]
        # un hijo nuevo que sale SPLIT crea a su vez hijos: repetir hasta estabilizar
        while True:
            nxt = pyr.exists(state)
            more = nxt & ~new_ex
            if not more.any():
                return nxt, (nxt & ~old_ex)
            state[more] = pyr.sample_states(fresh_prior, num, generator)[more]
            new_ex = nxt

    def resolve(t):
        """Sustituye los nodos recién nacidos por la predicción de la red, bajando
        de nivel hasta que no nazca ninguno más."""
        nonlocal state, ex, fresh
        for _ in range(pyr.depth + 1):
            if not fresh.any():
                break
            x0, p_cur = predict(t)
            if trace is not None:   # un recién nacido es ruido puro: cuenta como caliente
                trace.record(ex, p_cur.masked_fill(fresh, 0.0), fresh & (x0 != state), "nuevos")
            state = torch.where(fresh, x0, state)
            ex, fresh = refresh(ex)

    fresh = torch.zeros_like(ex)
    for t, s in zip(ts[:-1], ts[1:]):
        x0, p_cur = predict(t)
        jump = ex & (torch.rand(state.shape, device=dev, generator=generator) < (t - s) / t)
        if trace is not None:
            trace.record(ex, p_cur, jump & (x0 != state), "paso")
        state = torch.where(jump, x0, state)
        ex, fresh = refresh(ex)
        if resolve_fresh:
            resolve(max(s, 1e-3))
    # cierre: los nodos creados en el último paso aún son ruido; se resuelven aparte
    resolve(t_complete)
    stats = {"evals": evals, "tokens_por_muestra": tokens / num, "nodos_inicio": start_nodes,
             "nodos_final": ex.sum(1).float().mean().item(), "nodos_max": int(ex.sum(1).max()),
             "sin_resolver": int(fresh.sum())}
    return pyr.to_image(state, ex), stats
