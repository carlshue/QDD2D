"""
Transformer sobre nodos de quadtree. Cada nodo es un token con embeddings de
estado, nivel y posición; la red predice el estado {EMPTY, FULL, SPLIT} de los
nodos enmascarados.
"""
import math
import torch
import torch.nn as nn

from structures.binary_quadtree import NUM_STATES, SPLIT

MASK = NUM_STATES      # id de entrada para "nodo aún sin decidir"
PAD = NUM_STATES + 1   # id de entrada para relleno del lote


def local_allow(lq, yq, xq, lk, yk, xk, radius, hub):
    """Atención local sobre el árbol. Devuelve (B, Lq, Lk) bool: la consulta q puede mirar a la clave k si
    k está en su mismo nivel a distancia <= radius, es un antecesor suyo, es una hija directa, o una de
    las dos está en un nivel <= hub (celdas gruesas, que ven y son vistas por todas)."""
    li, lj = lq[:, :, None], lk[:, None, :]
    yi, yj, xi, xj = yq[:, :, None], yk[:, None, :], xq[:, :, None], xk[:, None, :]
    same = (li == lj) & ((yi - yj).abs() <= radius) & ((xi - xj).abs() <= radius)
    d = (li - lj).clamp(min=0)
    anc = (li > lj) & ((yi >> d) == yj) & ((xi >> d) == xj)
    child = (lj == li + 1) & ((yj >> 1) == yi) & ((xj >> 1) == xi)
    return same | anc | child | (li <= hub) | (lj <= hub)


class QuadtreeTransformer(nn.Module):
    def __init__(self, depth=5, dim=128, layers=4, heads=4, num_classes=10, dropout=0.0, use_time=False,
                 num_states=NUM_STATES, local=None):
        super().__init__()
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

    def forward(self, state, level, y, x, pad_mask, label=None, t=None):
        """state/level/y/x: (B, L) long; y, x en coordenadas de celda de su nivel.
        pad_mask: (B, L) bool, True en el relleno. Devuelve logits (B, L, 3)."""
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
        mask = None
        if self.local is not None:
            mask = ~local_allow(level, y, x, level, y, x, *self.local)
            mask = mask.repeat_interleave(self.heads, 0)
        h = self.encoder(h, mask=mask, src_key_padding_mask=pad_mask)
        logits = self.head(self.norm(h))
        if self.num_states != NUM_STATES:      # 'tile': sin estado SPLIT que prohibir
            return logits
        # un píxel (nivel más fino) no puede subdividirse
        finest = (level == self.depth)[..., None] & (torch.arange(NUM_STATES, device=state.device) == SPLIT)
        return logits.masked_fill(finest, float("-inf"))
