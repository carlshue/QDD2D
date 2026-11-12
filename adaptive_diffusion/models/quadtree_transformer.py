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


class QuadtreeTransformer(nn.Module):
    def __init__(self, depth=5, dim=128, layers=4, heads=4, num_classes=10, dropout=0.0, use_time=False):
        super().__init__()
        self.depth = depth
        side = 2 ** depth
        self.state_emb = nn.Embedding(NUM_STATES + 2, dim)
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
        self.head = nn.Linear(dim, NUM_STATES)

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
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        logits = self.head(self.norm(h))
        # un píxel (nivel más fino) no puede subdividirse
        finest = (level == self.depth)[..., None] & (torch.arange(NUM_STATES, device=state.device) == SPLIT)
        return logits.masked_fill(finest, float("-inf"))
