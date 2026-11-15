"""
Tokens de ocupación ('tile') generados en todos los niveles a la vez, con reenmascarado y
con recálculo sólo de las zonas calientes.

Representación densa: una celda por posición de cada nivel 0..depth-1 (1365 para depth 6).
El valor de una celda es el patrón de ocupación de sus 4 hijos (0..15) o T_MASK. Una celda
existe si su padre existe, está decidido y tiene puesto su bit; la raíz existe siempre.

Entrenamiento: cada nivel se tapa con una fracción propia; una celda cuyo antecesor está
tapado no aparece en la secuencia (no se sabe aún que existe).

Muestreo (al estilo de JoVox, arXiv 2611.10690): N pasos; en cada uno, cada nivel vuelve a
tapar sus celdas decididas más antiguas (lo que borra su subárbol) y destapa celdas al azar
hasta la fracción que le toca, kappa_l(k) = min(k / N_l, 1) ** rho, con N_l mayor en los
niveles finos.

Zonas calientes: una celda está caliente si está tapada o si algo cambió cerca de ella
(misma celda, vecinas, padre o hijas) en los últimos `heat` pasos. Sólo las calientes pasan
por la red; las frías conservan las activaciones de la última vez que se calcularon y sirven
de contexto. Con heat=None se recalcula todo (exacto).
"""
import math

import torch
import torch.nn.functional as F

from models.quadtree_transformer import block, local_allow
from models.tile_grow import T_MASK, T_PAD, T_STATES


class Grid:
    """Índices de la representación densa."""

    def __init__(self, depth, device="cpu"):
        self.depth = depth
        lv, ys, xs, par, bit, self.slices = [], [], [], [], [], []
        off = 0
        for l in range(depth):
            s = 2 ** l
            y, x = torch.meshgrid(torch.arange(s), torch.arange(s), indexing="ij")
            y, x = y.reshape(-1), x.reshape(-1)
            lv.append(torch.full((s * s,), l)); ys.append(y); xs.append(x)
            if l == 0:
                par.append(torch.zeros(1, dtype=torch.long)); bit.append(torch.zeros(1, dtype=torch.long))
            else:
                par.append(self.slices[-1].start + (y // 2) * (s // 2) + x // 2)
                bit.append(2 * (y % 2) + x % 2)
            self.slices.append(slice(off, off + s * s))
            off += s * s
        self.n = off
        self.level, self.y, self.x = (torch.cat(t).to(device) for t in (lv, ys, xs))
        self.parent, self.bit = torch.cat(par).to(device), torch.cat(bit).to(device)

    def existence(self, val):
        """(B, n) bool: celdas que existen dado lo decidido hasta ahora."""
        ex = torch.zeros_like(val, dtype=torch.bool)
        ex[:, 0] = True
        for l in range(1, self.depth):
            s = self.slices[l]
            p = self.parent[s]
            pv = val[:, p]
            ex[:, s] = ex[:, p] & (pv < T_STATES) & (((pv.clamp(max=T_STATES - 1) >> self.bit[s]) & 1) == 1)
        return ex

    def from_images(self, imgs):
        """imgs (B, S, S) en {0,1}. Devuelve (val, existe) del árbol verdadero."""
        occ = imgs.float()
        vals, exs = [], []
        for l in range(self.depth - 1, -1, -1):
            b, s2 = occ.shape[0], occ.shape[1]
            s = s2 // 2
            blk = occ.view(b, s, 2, s, 2).permute(0, 1, 3, 2, 4).reshape(b, s, s, 4)
            vals.append((blk * torch.tensor([1., 2., 4., 8.], device=occ.device)).sum(-1).long().reshape(b, -1))
            occ = blk.amax(-1)
            exs.append(occ.bool().reshape(b, -1))
        val, ex = torch.cat(vals[::-1], 1), torch.cat(exs[::-1], 1)
        ex[:, 0] = True
        return val, ex

    def to_image(self, val, ex):
        l = self.depth - 1
        s = 2 ** l
        v = val[:, self.slices[l]].clamp(max=T_STATES - 1).view(-1, s, s)
        e = ex[:, self.slices[l]].view(-1, s, s) & (val[:, self.slices[l]].view(-1, s, s) < T_STATES)
        img = torch.zeros(len(val), s, 2, s, 2, dtype=torch.uint8, device=val.device)
        for k in range(4):
            img[:, :, k // 2, :, k % 2] = (e & (((v >> k) & 1) == 1)).to(torch.uint8)
        return img.view(-1, 2 * s, 2 * s)

    def sequence(self, keep):
        """Orden que pone primero las celdas de `keep`. Devuelve (orden (B, L), relleno (B, L))."""
        n = int(keep.sum(1).max())
        order = torch.argsort((~keep).to(torch.uint8), dim=1, stable=True)[:, :n]
        return order, ~keep.gather(1, order)

    def neighbours(self, changed, radius=1):
        """Celdas cercanas a un cambio: misma zona, padre e hijas."""
        maps = [changed[:, s].float().view(-1, 1, 2 ** l, 2 ** l) for l, s in enumerate(self.slices)]
        out = []
        for l, m in enumerate(maps):
            a = F.max_pool2d(m, 2 * radius + 1, 1, radius) if radius > 0 else m
            if l + 1 < self.depth:
                a = torch.maximum(a, F.max_pool2d(maps[l + 1], 2))
            if l > 0:
                a = torch.maximum(a, F.interpolate(maps[l - 1], scale_factor=2, mode="nearest"))
            out.append(a.reshape(len(changed), -1) > 0)
        return torch.cat(out, 1)


def mask_rates(b, depth, device, generator=None):
    """Fracción tapada por (muestra, nivel). El 30 % de las veces independiente por nivel, el resto
    siguiendo un calendario como el del muestreo (niveles gruesos más avanzados)."""
    r = lambda *s: torch.rand(*s, device=device, generator=generator)
    ind = r(b, depth) * (r(b, depth) > 0.3)
    prog, rho = 1.1 * r(b, 1), 1 + 2 * r(b, 1)
    frac = (torch.arange(1, depth + 1, device=device) / depth)[None]
    sched = 1 - (prog / frac).clamp(max=1) ** rho
    return torch.where(r(b, 1) < 0.3, ind, sched)


def make_training_batch(grid, imgs, generator=None):
    val, ex = grid.from_images(imgs)
    b = len(val)
    m = mask_rates(b, grid.depth, val.device, generator)
    masked = torch.rand(b, grid.n, device=val.device, generator=generator) < m[:, grid.level]

    def present():
        pr = torch.zeros_like(ex)
        pr[:, 0] = True
        for l in range(1, grid.depth):
            s = grid.slices[l]
            p = grid.parent[s]
            pr[:, s] = ex[:, s] & pr[:, p] & ~masked[:, p]
        return pr
    pr = present()
    none = ~(pr & masked).any(1)
    if none.any():                       # al menos una celda que predecir: la raíz
        masked[none, 0] = True
        pr = present()
    order, pad = grid.sequence(pr)
    g = lambda t: t.gather(1, order) if t.dim() == 2 else t[order]
    mk = g(masked) & ~pad
    state = torch.where(mk, torch.full_like(order, T_MASK), g(val))
    state = torch.where(pad, torch.full_like(order, T_PAD), state)
    return {"state": state, "level": g(grid.level), "y": g(grid.y), "x": g(grid.x), "pad_mask": pad,
            "target": torch.where(mk, g(val), torch.full_like(order, -100))}


def loss_fn(model, batch, label=None):
    """Entropía cruzada media por (muestra, nivel), para que los niveles finos no dominen."""
    logits = model(batch["state"], batch["level"], batch["y"], batch["x"], batch["pad_mask"], label).float()
    tg = batch["target"]
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tg.reshape(-1), ignore_index=-100,
                         reduction="none").view(tg.shape)
    on = tg != -100
    group = torch.arange(len(tg), device=tg.device)[:, None] * model.depth + batch["level"]
    cnt = torch.zeros(len(tg) * model.depth, device=tg.device).scatter_add_(0, group[on], torch.ones_like(ce[on]))
    return (ce[on] / cnt[group[on]]).sum() / (cnt > 0).sum()


class CachedNet:
    """Evalúa la red sólo en las celdas pedidas, usando como contexto las activaciones guardadas
    de las demás. Si se piden todas las que existen, coincide con model.forward."""

    def __init__(self, model, grid, num, label=None):
        self.m, self.g = model, grid
        dev = grid.level.device
        cell = 2 ** (model.depth - grid.level)
        pos = model.level_emb(grid.level) + model.y_emb(2 * grid.y * cell + cell) + model.x_emb(2 * grid.x * cell + cell)
        if label is None:
            label = torch.full((num,), model.null_class, device=dev)
        self.base = pos[None] + model.class_emb(label)[:, None, :]
        self.cache = [None] + [torch.zeros(num, grid.n, model.dim, device=dev) for _ in model.encoder.layers]
        self.queries = 0      # celdas pasadas por la red
        self.pairs = 0        # pares consulta-contexto en la atención

    def __call__(self, val, ex, hot):
        m, g = self.m, self.g
        self.cache[0] = m.state_emb(val) + self.base
        oe, pe = g.sequence(ex)
        oh, ph = g.sequence(hot)
        rows = torch.arange(len(val), device=val.device)[:, None].expand_as(oh)[~ph]
        cols = oh[~ph]
        take = lambda t, o: t.gather(1, o[..., None].expand(-1, -1, t.shape[-1]))
        allow, npairs = None, hot.sum(1) * ex.sum(1)
        if m.local is not None:
            allow = local_allow(g.level[oh], g.y[oh], g.x[oh], g.level[oe], g.y[oe], g.x[oe], *m.local)
            npairs = (allow & ~ph[:, :, None] & ~pe[:, None, :]).sum((1, 2))
        for j, layer in enumerate(m.encoder.layers):
            h = block(layer, take(self.cache[j], oh), take(self.cache[j], oe), m.heads, allow, pe)
            self.cache[j + 1][rows, cols] = h[~ph].to(self.cache[j + 1].dtype)
        logits = torch.full((len(val), g.n, m.num_states), float("-inf"), device=val.device)
        logits[rows, cols] = m.head(m.norm(self.cache[-1][rows, cols])).float()
        self.queries += int(hot.sum())
        self.pairs += int(npairs.sum())
        return logits


def _pick(score, cand, count):
    """Por muestra, las `count` celdas candidatas de mayor puntuación. score/cand (B, n), count (B,)."""
    sc = torch.where(cand, score, torch.full_like(score, float("-inf")))
    idx = sc.argsort(1, descending=True)
    sel = torch.arange(sc.shape[1], device=sc.device)[None] < torch.minimum(count, cand.sum(1))[:, None]
    return torch.zeros_like(cand).scatter_(1, idx, sel)


@torch.no_grad()
def sample(model, num, steps=32, label=None, temperature=1.0, device="cpu", generator=None,
           remask="oldest", rho=2.0, heat=None, radius=1, stats=None, back=1):
    """Devuelve (imágenes (num, S, S) uint8, estadísticas).
    remask: 'none', 'oldest' (las decididas hace más tiempo) o 'conf' (las de menor probabilidad).
    back: cuántos pasos de calendario retrocede cada nivel al reenmascarar (más = más revisión).
    heat: None recalcula todas las celdas en cada paso; un entero k recalcula sólo las tapadas y las
          que tuvieron un cambio cerca en los últimos k pasos."""
    grid = Grid(model.depth, device)
    D = grid.depth
    if label is not None:
        label = label.to(device)
    net = CachedNet(model, grid, num, label)
    val = torch.full((num, grid.n), T_MASK, dtype=torch.long, device=device)
    ex = grid.existence(val)
    age = torch.zeros(num, grid.n, device=device)
    conf = torch.zeros(num, grid.n, device=device)
    heat_left = torch.zeros(num, grid.n, dtype=torch.long, device=device)
    n_l = [max(1, math.ceil(steps * (l + 1) / D)) for l in range(D)]
    kappa = lambda l, k: min(k / n_l[l], 1.0) ** rho
    full = remasked = evals = 0
    rnd = lambda: torch.rand(num, grid.n, device=device, generator=generator)
    for k in range(1, steps + D + 3):
        extra = k > steps
        changed = torch.zeros_like(ex)
        if remask != "none" and not extra and k > 1:
            for l, s in enumerate(grid.slices):
                done = ex[:, s] & (val[:, s] != T_MASK)
                target = torch.ceil(kappa(l, max(k - 1 - back, 0)) * ex[:, s].sum(1).float()).long()
                score = -age[:, s] if remask == "oldest" else -conf[:, s]
                sel = _pick(score + 1e-3 * rnd()[:, s], done, (done.sum(1) - target).clamp(min=0))
                val[:, s][sel] = T_MASK
                changed[:, s] |= sel
                remasked += int(sel.sum())
            ex = grid.existence(val)
            val[~ex] = T_MASK
        masked = ex & (val == T_MASK)
        if not masked.any():
            if extra:
                break
            continue
        hot = ex if heat is None else masked | (ex & (heat_left > 0))
        logits = net(val, ex, hot)
        evals += 1
        full += int(ex.sum())
        logits[..., 0] = float("-inf")           # una celda que existe tiene algún hijo ocupado
        probs = F.softmax(logits[masked] / temperature, -1)
        new = torch.multinomial(probs, 1, generator=generator)[:, 0]
        cand_val = torch.full_like(val, T_MASK)
        cand_val[masked] = new
        cand_conf = torch.zeros_like(conf)
        cand_conf[masked] = probs.gather(1, new[:, None])[:, 0]
        r = rnd()
        for l, s in enumerate(grid.slices):
            mk = masked[:, s]
            n_ex = ex[:, s].sum(1)
            if extra:
                count = mk.sum(1)
            else:
                count = (torch.ceil(kappa(l, k) * n_ex.float()).long() - (n_ex - mk.sum(1))).clamp(min=0)
            sel = _pick(r[:, s], mk, count)
            val[:, s] = torch.where(sel, cand_val[:, s], val[:, s])
            age[:, s] = torch.where(sel, torch.full_like(age[:, s], float(k)), age[:, s])
            conf[:, s] = torch.where(sel, cand_conf[:, s], conf[:, s])
            changed[:, s] |= sel
        new_ex = grid.existence(val)
        changed |= new_ex & ~ex
        ex = new_ex
        val[~ex] = T_MASK
        heat_left = (heat_left - 1).clamp(min=0)
        if heat:
            heat_left[grid.neighbours(changed, radius)] = heat
    st = {"evals": evals, "tokens_por_muestra": net.queries / num, "tokens_sin_cache": full / num,
          "pares_por_muestra": net.pairs / num, "reenmascarados": remasked / num,
          "nodos_final": float(ex.sum()) / num}
    if stats is not None:
        stats.update(st)
    return grid.to_image(val, ex), st


@torch.no_grad()
def sample_levels(model, num, steps_per_level=8, label=None, temperature=1.0, device="cpu", generator=None,
                  heat=None, radius=1, order="random", no_zero=False, stats=None, lazy=0):
    """Muestreo nivel a nivel de un modelo 'tile' (models/tile_grow.py), en denso y con zonas calientes.
    Con heat=None, order='random' y no_zero=False reproduce tile_grow.sample.
    steps_per_level y temperature pueden ser listas, una entrada por nivel.
    order: 'random' destapa celdas al azar; 'conf' destapa primero las de predicción más segura.
    lazy (sólo con heat y order='random'): las celdas que se van a destapar se eligen antes de evaluar la
          red y sólo ésas (más las calientes) pasan por ella; las demás tapadas quedan como contexto con
          lo último calculado. lazy=1: nunca se calculan antes; lazy=2: se calculan una vez al nacer."""
    grid = Grid(model.depth, device)
    D = grid.depth
    if label is not None:
        label = label.to(device)
    net = CachedNet(model, grid, num, label)
    spl = steps_per_level if isinstance(steps_per_level, (list, tuple)) else [steps_per_level] * D
    temps = temperature if isinstance(temperature, (list, tuple)) else [temperature] * D
    val = torch.full((num, grid.n), T_MASK, dtype=torch.long, device=device)
    heat_left = torch.zeros(num, grid.n, dtype=torch.long, device=device)
    upto = lambda l: (grid.level <= l)[None]
    ex = grid.existence(val) & upto(0)
    full = evals = 0
    for l in range(D):
        for step in range(spl[l]):
            masked = ex & (val == T_MASK)
            if not masked.any():
                break
            keep = math.cos(math.pi / 2 * (step + 1) / spl[l]) if step < spl[l] - 1 else 0.0
            n_m = masked.sum(1)
            count = n_m - (keep * n_m.float()).long()
            score = torch.rand(num, grid.n, device=device, generator=generator)
            pre = _pick(score, masked, count) if (lazy and heat is not None and order == "random") else None
            if heat is None:
                hot = ex
            elif pre is None or (lazy == 2 and step == 0):
                hot = masked | (ex & (heat_left > 0))
            else:
                hot = pre | (ex & (heat_left > 0))
            logits = net(val, ex, hot)
            evals += 1
            full += int(ex.sum())
            if no_zero:
                logits[..., 0] = float("-inf")
            draw = masked if pre is None else pre
            probs = F.softmax(logits[draw] / temps[l], -1)
            new = torch.multinomial(probs, 1, generator=generator)[:, 0]
            cand = torch.full_like(val, T_MASK)
            cand[draw] = new
            if order == "conf":
                score = torch.zeros_like(score)
                score[masked] = probs.gather(1, new[:, None])[:, 0]
            sel = pre if pre is not None else _pick(score, masked, count)
            val = torch.where(sel, cand, val)
            heat_left = (heat_left - 1).clamp(min=0)
            if heat:
                heat_left[grid.neighbours(sel, radius)] = heat
        if l < D - 1:
            new_ex = grid.existence(val) & upto(l + 1)
            if heat:
                heat_left[grid.neighbours(new_ex & ~ex, radius)] = heat
            ex = new_ex
    ex = grid.existence(val)
    st = {"evals": evals, "tokens_por_muestra": net.queries / num, "tokens_sin_cache": full / num,
          "pares_por_muestra": net.pairs / num, "nodos_final": float(ex.sum()) / num}
    if stats is not None:
        stats.update(st)
    return grid.to_image(val, ex), st
