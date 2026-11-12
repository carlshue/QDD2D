"""
Pirámide densa de estados de quadtree, en PyTorch y por lotes.

Un árbol se guarda como un vector de `size` celdas (todas las celdas posibles
de todos los niveles: 1 + 4 + 16 + ...). Una celda "existe" si su padre existe
y está en SPLIT. Esta forma densa permite corromper, editar y decodificar
árboles enteros en GPU sin bucles por muestra.

También define las distribuciones de "árbol aleatorio" (el ruido).
"""
import torch

from structures.binary_quadtree import EMPTY, FULL, SPLIT


class Pyramid:
    def __init__(self, depth, device="cpu"):
        self.depth = depth
        self.side = 2 ** depth
        self.offsets = [(4 ** l - 1) // 3 for l in range(depth + 2)]
        self.size = self.offsets[depth + 1]
        lv, ys, xs, par = [], [], [], []
        for l in range(depth + 1):
            s = 2 ** l
            y, x = torch.meshgrid(torch.arange(s), torch.arange(s), indexing="ij")
            y, x = y.reshape(-1), x.reshape(-1)
            lv.append(torch.full_like(y, l)); ys.append(y); xs.append(x)
            par.append(torch.zeros_like(y) if l == 0 else self.offsets[l - 1] + (y // 2) * (s // 2) + x // 2)
        self.level = torch.cat(lv).to(device)
        self.y = torch.cat(ys).to(device)
        self.x = torch.cat(xs).to(device)
        self.parent = torch.cat(par).to(device)
        self.device = device

    def _slice(self, l):
        return slice(self.offsets[l], self.offsets[l + 1])

    # ---------- imagen <-> árbol ----------
    def from_images(self, imgs):
        """imgs (N, S, S) en {0,1} -> estado verdadero de TODAS las celdas (N, size):
        EMPTY/FULL si la región es uniforme, SPLIT si es mixta."""
        imgs = imgs.to(torch.uint8)
        n = imgs.shape[0]
        out = []
        for l in range(self.depth + 1):
            s, k = 2 ** l, self.side >> l
            b = imgs.reshape(n, s, k, s, k)
            mx, mn = b.amax((2, 4)), b.amin((2, 4))
            st = torch.where(mx == 0, EMPTY, torch.where(mn == 1, FULL, SPLIT))
            out.append(st.reshape(n, -1))
        return torch.cat(out, 1).to(torch.uint8)

    def exists(self, state):
        """Qué celdas existen dado el estado de todas (de arriba abajo)."""
        e = torch.zeros_like(state, dtype=torch.bool)
        e[:, 0] = True
        for l in range(1, self.depth + 1):
            sl = self._slice(l)
            p = self.parent[sl]
            e[:, sl] = e[:, p] & (state[:, p] == SPLIT)
        return e

    def to_image(self, state, exists):
        n = state.shape[0]
        img = torch.zeros(n, self.side, self.side, dtype=torch.bool, device=state.device)
        full = exists & (state == FULL)
        for l in range(self.depth + 1):
            s, k = 2 ** l, self.side >> l
            m = full[:, self._slice(l)].reshape(n, s, s)
            img |= m.repeat_interleave(k, 1).repeat_interleave(k, 2)
        return img

    # ---------- ruido: árboles aleatorios ----------
    def prior_uniform(self, p_split=0.667):
        """Subdivisión al azar: cada nodo se subdivide con prob. p_split; si no,
        es hoja vacía o llena a partes iguales. No sabe nada de los datos."""
        pr = torch.tensor([(1 - p_split) / 2, (1 - p_split) / 2, p_split]).repeat(self.size, 1)
        pr[self._slice(self.depth)] = torch.tensor([0.5, 0.5, 0.0])
        return pr.to(self.device)

    def prior_from_data(self, s0, mode="cell", smoothing=1.0):
        """Ruido informado por los datos. 'level': frecuencia de cada estado por
        nivel. 'cell': frecuencia por celda (sabe, p. ej., que las esquinas
        suelen estar vacías). Siempre condicionado a que la celda exista."""
        counts = torch.zeros(self.size, 3, device=s0.device)
        for chunk in s0.split(8192):
            ex = self.exists(chunk)
            for s in (EMPTY, FULL, SPLIT):
                counts[:, s] += ((chunk == s) & ex).sum(0)
        level_p = torch.zeros_like(counts)
        for l in range(self.depth + 1):
            sl = self._slice(l)
            tot = counts[sl].sum(0)
            level_p[sl] = tot / tot.sum()
        if mode == "level":
            return level_p
        pr = counts + smoothing * level_p
        return pr / pr.sum(-1, keepdim=True)

    def sample_states(self, prior, n, generator=None):
        u = torch.rand(n, self.size, device=prior.device, generator=generator)
        cdf = prior.cumsum(-1)
        return (u[..., None] > cdf[None, :, :2]).sum(-1).to(torch.uint8)

    def corrupt(self, s0, t, prior, generator=None):
        """Proceso de ruido: cada celda conserva su estado real con prob. 1-t y,
        si no, toma un estado al azar del ruido. Devuelve (estado, existe)."""
        noise = torch.rand(s0.shape, device=s0.device, generator=generator) < t[:, None]
        state = torch.where(noise, self.sample_states(prior, s0.shape[0], generator), s0)
        return state, self.exists(state)

    def to_sequence(self, exists):
        """Índices de las celdas existentes, rellenados a la longitud máxima del lote."""
        n = int(exists.sum(1).max())
        order = torch.argsort(exists.to(torch.uint8), dim=1, descending=True, stable=True)[:, :n]
        return order, ~exists.gather(1, order)
