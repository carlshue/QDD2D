"""Voxelización de mallas de triángulos a rejillas binarias."""
import numpy as np


def voxelize_surface(v, f, res, points=None, rng=None):
    """Ocupación de superficie (res³, bool): muestrea puntos en los triángulos y los reparte en vóxeles.
    La malla se centra y se escala a la caja unidad (lado mayor = 0,98). Devuelve None si la malla no vale."""
    rng = rng or np.random.default_rng(0)
    points = points or 300 * res * res
    v = np.asarray(v, np.float64)
    lo, hi = v.min(0), v.max(0)
    if not np.isfinite(lo).all() or (hi - lo).max() <= 0:
        return None
    v = (v - (lo + hi) / 2) / (hi - lo).max() * 0.98 + 0.5
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    if not np.isfinite(area).all() or area.sum() <= 0:
        return None
    tri = rng.choice(len(f), points, p=area / area.sum())
    u, w = rng.random(points), rng.random(points)
    flip = u + w > 1
    u[flip], w[flip] = 1 - u[flip], 1 - w[flip]
    p = a[tri] + u[:, None] * (b[tri] - a[tri]) + w[:, None] * (c[tri] - a[tri])
    idx = np.clip((p * res).astype(np.int64), 0, res - 1)
    vox = np.zeros((res, res, res), bool)
    vox[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return vox


def fill_solid(surface):
    """Rellena el interior: todo lo que no se alcanza desde el borde por celdas vacías."""
    free = ~surface
    out = np.zeros_like(surface)
    out[0], out[-1], out[:, 0], out[:, -1], out[:, :, 0], out[:, :, -1] = True, True, True, True, True, True
    out &= free
    while True:
        grow = out.copy()
        grow[1:] |= out[:-1]; grow[:-1] |= out[1:]
        grow[:, 1:] |= out[:, :-1]; grow[:, :-1] |= out[:, 1:]
        grow[:, :, 1:] |= out[:, :, :-1]; grow[:, :, :-1] |= out[:, :, 1:]
        grow &= free
        if (grow == out).all():
            return ~out
        out = grow
