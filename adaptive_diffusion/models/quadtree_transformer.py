"""
Transformer sobre nodos de quadtree. Cada nodo es un token con embeddings de
estado, nivel y posición; la red predice el estado {EMPTY, FULL, SPLIT} de los
nodos enmascarados.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from structures.binary_quadtree import NUM_STATES, SPLIT

MASK = NUM_STATES      # id de entrada para "nodo aún sin decidir"
PAD = NUM_STATES + 1   # id de entrada para relleno del lote


def local_allow(lq, yq, xq, lk, yk, xk, radius, hub):
    """Atención local sobre el árbol. Devuelve (B, Lq, Lk) bool: la consulta q puede mirar a la clave k si
    k está en su mismo nivel a distancia <= radius, es un antecesor suyo, es una hija directa, o una de
    las dos está en un nivel <= hub (celdas gruesas, que ven y son vistas por todas)."""
    lq, yq, xq, lk, yk, xk = (t.to(torch.int16) for t in (lq, yq, xq, lk, yk, xk))     # (B, Lq, Lk) pesa: enteros cortos
    li, lj = lq[:, :, None], lk[:, None, :]
    yi, yj, xi, xj = yq[:, :, None], yk[:, None, :], xq[:, :, None], xk[:, None, :]
    same = (li == lj) & ((yi - yj).abs() <= radius) & ((xi - xj).abs() <= radius)
    d = (li - lj).clamp(min=0)
    anc = (li > lj) & ((yi >> d) == yj) & ((xi >> d) == xj)
    child = (lj == li + 1) & ((yj >> 1) == yi) & ((xj >> 1) == xi)
    return same | anc | child | (li <= hub) | (lj <= hub)


def block(layer, q, kv, heads, allow=None, key_pad=None):
    """Una capa del codificador (pre-norma) con consultas q (B, Lq, d) y contexto kv (B, Lk, d).
    allow (B, Lq, Lk) bool: pares permitidos; key_pad (B, Lk) bool: relleno. Mismos pesos que
    nn.TransformerEncoderLayer, pero la máscara no se copia por cabeza ni se pasa a flotante."""
    at = layer.self_attn
    b, lq, d = q.shape
    w, bias = at.in_proj_weight, at.in_proj_bias
    qn, kn = layer.norm1(q), layer.norm1(kv)
    split = lambda t: t.view(b, -1, heads, d // heads).transpose(1, 2)
    qq = split(F.linear(qn, w[:d], bias[:d]))
    kk = split(F.linear(kn, w[d:2 * d], bias[d:2 * d]))
    vv = split(F.linear(kn, w[2 * d:], bias[2 * d:]))
    mask = None
    if allow is not None or key_pad is not None:
        mask = torch.ones(b, lq, kv.shape[1], dtype=torch.bool, device=q.device) if allow is None else allow
        if key_pad is not None:
            mask = mask & ~key_pad[:, None, :]
        mask = (mask | ~mask.any(-1, keepdim=True))[:, None]      # una fila sin nada permitido (relleno) daría NaN
    o = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=mask)
    h = q + at.out_proj(o.transpose(1, 2).reshape(b, lq, d))
    return h + layer.linear2(layer.activation(layer.linear1(layer.norm2(h))))


class QuadtreeTransformer(nn.Module):
    def __init__(self, depth=5, dim=128, layers=4, heads=4, num_classes=10, dropout=0.0, use_time=False,
                 num_states=NUM_STATES, local=None, ctx=False):
        super().__init__()
        # ctx: cada token recibe además el estado de su padre y qué vecinas de su nivel existen (8 bits)
        self.parent_emb = nn.Embedding(num_states + 3, dim) if ctx else None
        self.nb_emb = nn.Embedding(256, dim) if ctx else None
        self.register_buffer("level_off", torch.tensor([(4 ** l - 1) // 3 for l in range(depth + 2)]), persistent=False)
        self.local = local                # None = atención global; (radius, hub) = atención local (local_allow)
        self.heads = heads
        self.depth = depth
        side = 2 ** depth
        self.num_states = num_states      # ids de entrada extra: num_states = MASK, num_states + 1 = PAD
        self.state_emb = nn.Embedding(num_states + 2, dim)
        self.level_emb = nn.Embedding(depth + 1, dim)
        # posición = centro de la celda a resolución 2x de píxel (distingue niveles)
        self.y_emb = nn.Embedding(2 * side, dim)
        self.x_emb = nn.Embedding(2 * side, dim)
        self.class_emb = nn.Embedding(num_classes + 1, dim)  # último id = sin condición
        self.null_class = num_classes
        self.dim = dim
        # nivel de ruido t: sólo lo usa la difusión por edición
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)) if use_time else None
        block = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_states)

    def embed(self, state, level, y, x, pad_mask, label=None, t=None):
        """Entrada de la primera capa: estado + nivel + posición + clase (+ tiempo, + contexto local)."""
        cell = 2 ** (self.depth - level)       # lado de la celda en píxeles
        cy = 2 * y * cell + cell               # centro, en medios píxeles
        cx = 2 * x * cell + cell
        h = self.state_emb(state) + self.level_emb(level) + self.y_emb(cy) + self.x_emb(cx)
        if label is None:
            label = torch.full((state.shape[0],), self.null_class, device=state.device)
        h = h + self.class_emb(label)[:, None, :]
        if self.time_mlp is not None and t is not None:
            half = self.dim // 2
            freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
            ang = 1000.0 * t[:, None] * freqs[None]
            h = h + self.time_mlp(torch.cat([ang.sin(), ang.cos()], -1))[:, None, :]
        if self.parent_emb is not None:
            n = int(self.level_off[-1])                    # tabla densa nivel/y/x -> estado; n = hueco
            none = self.num_states + 2
            side = 1 << level
            flat = torch.where(pad_mask, torch.full_like(level, n), self.level_off[level] + y * side + x)
            table = torch.full((state.shape[0], n + 1), none, dtype=torch.long, device=state.device)
            table.scatter_(1, flat, state)
            table[:, n] = none
            lp = (level - 1).clamp(min=0)
            par = torch.where(level > 0, self.level_off[lp] + (y >> 1) * (1 << lp) + (x >> 1), torch.full_like(level, n))
            h = h + self.parent_emb(table.gather(1, par))
            bits = torch.zeros_like(level)
            k = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    yy, xx = y + dy, x + dx
                    ok = (yy >= 0) & (yy < side) & (xx >= 0) & (xx < side)
                    idx = torch.where(ok, self.level_off[level] + yy.clamp(min=0) * side + xx.clamp(min=0), torch.full_like(level, n))
                    bits = bits | ((table.gather(1, idx.clamp(max=n)) != none).long() << k)
                    k += 1
            h = h + self.nb_emb(bits)
        return h

    def forward(self, state, level, y, x, pad_mask, label=None, t=None):
        """state/level/y/x: (B, L) long; y, x en coordenadas de celda de su nivel.
        pad_mask: (B, L) bool, True en el relleno. Devuelve logits (B, L, 3)."""
        h = self.embed(state, level, y, x, pad_mask, label, t)
        if self.local is not None:      # atención local: capas a mano, con la máscara compartida entre cabezas
            allow = local_allow(level, y, x, level, y, x, *self.local)
            for layer in self.encoder.layers:
                h = block(layer, h, h, self.heads, allow, pad_mask)
        else:
            h = self.encoder(h, src_key_padding_mask=pad_mask)
        logits = self.head(self.norm(h))
        if self.num_states != NUM_STATES:      # 'tile': sin estado SPLIT que prohibir
            return logits
        # un píxel (nivel más fino) no puede subdividirse
        finest = (level == self.depth)[..., None] & (torch.arange(NUM_STATES, device=state.device) == SPLIT)
        return logits.masked_fill(finest, float("-inf"))
