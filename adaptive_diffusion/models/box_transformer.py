"""
Transformer sobre árboles de cajas 3D (octree u omnitree, ver structures/boxtree.py).

Cada token es un nodo mixto. La red recibe su caja (nivel e índice por eje, profundidad en el árbol) y,
si ya está decidido, sus tres piezas: corte, ocupación de los hijos y cuáles de ellos son hojas llenas.
Para los tokens tapados predice las piezas en cadena, con una sola pasada del transformer:

    p(corte) · p(ocupación | corte) · p(lleno | corte, ocupación)

Las dos últimas cabezas reciben la salida del transformer más un embedding de las piezas anteriores.
Los valores imposibles se excluyen: no se parte un eje agotado, la ocupación no puede ser cero ni usar
hijos que no existen, y `lleno` es un subconjunto de la ocupación.

Con `predict_cut=False` el corte es siempre el de los tres ejes (octree) y con `predict_full=False` no
hay hojas llenas por encima del vóxel (`tile`): es el mismo modelo con menos cabezas.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CUT_MASK, CUT_PAD = 8, 9            # el corte vale 1..7 (máscara de ejes)
OCC_MASK, OCC_PAD = 256, 257        # ocupación y lleno valen 0..255 (máscara de hijos)
ALL_AXES = 7


def popcount3(c):
    return (c & 1) + ((c >> 1) & 1) + ((c >> 2) & 1)


class BoxTransformer(nn.Module):
    def __init__(self, depth, dim=128, layers=4, heads=4, num_classes=10, dropout=0.0,
                 predict_cut=True, predict_full=True):
        super().__init__()
        self.depth, self.dim = depth, dim                    # depth = D: la rejilla es de lado 2^D
        self.predict_cut, self.predict_full = predict_cut, predict_full
        side = 2 ** depth
        self.cut_emb = nn.Embedding(10, dim)
        self.occ_emb = nn.Embedding(258, dim)
        self.full_emb = nn.Embedding(258, dim)
        self.lev_emb = nn.ModuleList([nn.Embedding(depth + 1, dim) for _ in range(3)])
        self.pos_emb = nn.ModuleList([nn.Embedding(2 * side + 1, dim) for _ in range(3)])   # centro, en medios vóxeles
        self.tree_emb = nn.Embedding(3 * depth + 2, dim)                                    # profundidad en el árbol
        self.class_emb = nn.Embedding(num_classes + 1, dim)
        self.null_class = num_classes
        block = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.cut_head = nn.Linear(dim, 8)
        self.cut_cond = nn.Embedding(8, dim)
        self.occ_cond = nn.Embedding(256, dim)
        mlp = lambda: nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 256))
        self.occ_head, self.full_head = mlp(), mlp()
        codes = torch.arange(256)
        nchild = 2 ** popcount3(torch.arange(8))                                 # hijos por corte
        self.register_buffer("occ_ok", (codes[None] >= 1) & (codes[None] < 2 ** nchild[:, None]), persistent=False)
        self.register_buffer("all_children", 2 ** nchild - 1, persistent=False)

    def trunk(self, b, label=None):
        """b: dict con depth (B, L), lev e idx (B, L, 3), cut, occ, full (B, L), pad (B, L) bool. Devuelve (B, L, dim)."""
        h = self.cut_emb(b["cut"]) + self.occ_emb(b["occ"]) + self.full_emb(b["full"]) + self.tree_emb(b["depth"])
        half = 2 ** (self.depth - b["lev"])                  # medio lado de la caja, en medios vóxeles... por eje
        center = (2 * b["idx"] + 1) * half
        for a in range(3):
            h = h + self.lev_emb[a](b["lev"][..., a]) + self.pos_emb[a](center[..., a])
        if label is None:
            label = torch.full((h.shape[0],), self.null_class, device=h.device)
        h = h + self.class_emb(label)[:, None, :]
        return self.norm(self.encoder(h, src_key_padding_mask=b["pad"]))

    # Las cabezas trabajan sobre los tokens elegidos: h (M, dim), lev (M, 3), cut y occ (M,)
    def cut_logits(self, h, lev):
        free = ((lev < self.depth).long() * torch.tensor([1, 2, 4], device=lev.device)).sum(-1)    # ejes aún partibles
        c = torch.arange(8, device=h.device)[None]
        ok = (c > 0) & ((c & ~free[:, None]) == 0)
        return self.cut_head(h).float().masked_fill(~ok, float("-inf"))

    def occ_logits(self, h, cut):
        return self.occ_head(h + self.cut_cond(cut)).float().masked_fill(~self.occ_ok[cut], float("-inf"))

    def full_logits(self, h, cut, occ):
        f = torch.arange(256, device=h.device)[None]
        ok = (f & ~occ[:, None]) == 0                                             # subconjunto de la ocupación
        ok &= ~((occ == self.all_children[cut])[:, None] & (f == occ[:, None]))   # todos llenos: el nodo no sería mixto
        return self.full_head(h + self.cut_cond(cut) + self.occ_cond(occ)).float().masked_fill(~ok, float("-inf"))

    def unit_children(self, lev, cut):
        """True si los hijos del nodo son vóxeles: entonces `lleno` = ocupación y no se predice."""
        bits = torch.stack([(cut >> a) & 1 for a in range(3)], -1)
        return ((lev + bits) == self.depth).all(-1)

    def loss(self, b, label=None):
        """Entropía cruzada de las piezas de los tokens tapados (b["masked"]), en nats por token tapado."""
        h = self.trunk(b, label)[b["masked"]]
        lev, cut, occ, full = (b["target_" + k][b["masked"]] for k in ("lev", "cut", "occ", "full"))
        n = max(len(h), 1)
        parts = {}
        if self.predict_cut:
            parts["corte"] = F.cross_entropy(self.cut_logits(h, lev), cut, reduction="sum") / n
        parts["ocupacion"] = F.cross_entropy(self.occ_logits(h, cut), occ, reduction="sum") / n
        if self.predict_full:
            free = ~self.unit_children(lev, cut)
            if free.any():
                parts["lleno"] = F.cross_entropy(self.full_logits(h[free], cut[free], occ[free]), full[free],
                                                 reduction="sum") / n
        return sum(parts.values()), {k: v.detach() for k, v in parts.items()}

    @torch.no_grad()
    def sample_pieces(self, h, lev, temperature=1.0, generator=None):
        """Sortea corte, ocupación y lleno para los tokens con salida h (M, dim). Devuelve tres tensores (M,)."""
        draw = lambda logits: torch.multinomial(F.softmax(logits / temperature, -1), 1, generator=generator)[:, 0]
        cut = draw(self.cut_logits(h, lev)) if self.predict_cut else torch.full((len(h),), ALL_AXES, device=h.device)
        occ = draw(self.occ_logits(h, cut))
        unit = self.unit_children(lev, cut)
        full = torch.where(unit, occ, torch.zeros_like(occ))
        if self.predict_full and (~unit).any():
            full[~unit] = draw(self.full_logits(h[~unit], cut[~unit], occ[~unit]))
        return cut, occ, full
