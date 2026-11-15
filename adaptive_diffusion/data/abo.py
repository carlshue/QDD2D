"""
ABO (Amazon Berkeley Objects) voxelizado: formas 3D de productos, sobre todo muebles.

    load_abo(res=32, train=True) -> (vox (N, res, res, res) uint8 en {0,1}, labels (N,) int64)

Ocupación de superficie: un vóxel está ocupado si la superficie de la malla pasa por él. Las formas
vienen en su orientación de diseño (rectas, alineadas con los ejes; el eje 1 es la vertical).

El fichero `data/abo/abo{res}.npz` se construye con

    python -m data.abo --res 32

que descarga sólo la geometría de cada modelo (por rangos HTTP, sin texturas) y la voxeliza.
`data/abo/categories.json` (id -> tipo de producto) sale de los metadatos de ABO.

Datos originales: Amazon Berkeley Objects, https://amazon-berkeley-objects.s3.amazonaws.com/index.html
(Collins et al., CVPR 2022), licencia CC BY-NC 4.0. Los ficheros de este directorio son derivados suyos.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import struct
import urllib.request

import numpy as np

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "abo")
BASE = "https://amazon-berkeley-objects.s3.amazonaws.com/3dmodels/"

# 10 clases; los tipos que no están aquí van a "otros". Las categorías planas (alfombras, cuadros,
# espejos) se excluyen por defecto: son losas y no dicen nada de la generación.
CLASSES = ("silla", "sofa", "mesa", "decoracion", "lampara", "taburete", "otomana", "cojin", "cama", "otros")
_TYPE_TO_CLASS = {"CHAIR": 0, "SOFA": 1, "TABLE": 2, "DESK": 2, "HOME_FURNITURE_AND_DECOR": 3, "LAMP": 4,
                  "LIGHT_FIXTURE": 4, "STOOL_SEATING": 5, "OTTOMAN": 6, "PILLOW": 7, "HEADBOARD": 8, "BED": 8,
                  "BED_FRAME": 8}
FLAT_TYPES = ("RUG", "WALL_ART", "HOME_MIRROR")
_CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def is_test(model_id):
    """Reparto fijo entrenamiento/prueba (90/10) por el identificador del modelo."""
    return int(hashlib.md5(model_id.encode()).hexdigest(), 16) % 10 == 0


def load_abo(res=32, train=True, flat=False):
    path = os.path.join(HERE, f"abo{res}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Falta {path}. Constrúyelo con:  python -m data.abo --res {res}")
    d = np.load(path)
    ids, types = d["ids"], d["types"]
    vox = np.unpackbits(d["bits"], axis=1)[:, : res ** 3].reshape(-1, res, res, res)
    labels = np.array([_TYPE_TO_CLASS.get(t, len(CLASSES) - 1) for t in types], np.int64)
    keep = np.array([is_test(i) != train for i in ids])
    if not flat:
        keep &= ~np.isin(types, FLAT_TYPES)
    return vox[keep], labels[keep]


# ---------------------------------------------------------------- construcción del fichero

def _range(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
    for attempt in range(3):
        try:
            return urllib.request.urlopen(req, timeout=60).read()
        except Exception:
            if attempt == 2:
                raise


def _node_matrix(n):
    if "matrix" in n:
        return np.array(n["matrix"], float).reshape(4, 4).T
    m = np.eye(4)
    if "scale" in n:
        m = np.diag(list(n["scale"]) + [1.0]) @ m
    if "rotation" in n:
        x, y, z, w = n["rotation"]
        r = np.eye(4)
        r[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
        m = r @ m
    if "translation" in n:
        t = np.eye(4)
        t[:3, 3] = n["translation"]
        m = t @ m
    return m


def fetch_geometry(glb_path):
    """Vértices y caras de un .glb de ABO, leyendo sólo la cabecera y los tramos de geometría."""
    url = BASE + "original/" + glb_path
    magic, _, _, jl, _ = struct.unpack("<4sIIII", _range(url, 0, 19))
    assert magic == b"glTF"
    g = json.loads(_range(url, 20, 20 + jl - 1))
    bin0 = 20 + jl + 8

    def accessor(i):
        a = g["accessors"][i]
        bv = g["bufferViews"][a["bufferView"]]
        dt, nc = np.dtype(_CT[a["componentType"]]), _NC[a["type"]]
        start = bin0 + bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = bv.get("byteStride") or dt.itemsize * nc
        raw = _range(url, start, start + stride * (a["count"] - 1) + dt.itemsize * nc - 1)
        if stride == dt.itemsize * nc:
            return np.frombuffer(raw, dt).reshape(a["count"], nc)
        pick = (np.arange(a["count"])[:, None] * stride + np.arange(dt.itemsize * nc)[None]).reshape(-1)
        return np.frombuffer(np.frombuffer(raw, np.uint8)[pick].tobytes(), dt).reshape(a["count"], nc)

    verts, faces, off = [], [], 0

    def walk(ni, parent):
        nonlocal off
        n = g["nodes"][ni]
        m = parent @ _node_matrix(n)
        if "mesh" in n:
            for p in g["meshes"][n["mesh"]]["primitives"]:
                if p.get("mode", 4) != 4:
                    continue
                v = accessor(p["attributes"]["POSITION"]).astype(np.float64)
                f = accessor(p["indices"]).reshape(-1, 3).astype(np.int64) if "indices" in p \
                    else np.arange(len(v)).reshape(-1, 3)
                verts.append(v @ m[:3, :3].T + m[:3, 3])
                faces.append(f + off)
                off += len(v)
        for c in n.get("children", []):
            walk(c, m)

    for root in g["scenes"][g.get("scene", 0)]["nodes"]:
        walk(root, np.eye(4))
    return np.concatenate(verts).astype(np.float32), np.concatenate(faces).astype(np.int32)


def _one(job):
    """Geometría (de disco o descargada) -> vóxeles empaquetados. Devuelve (id, bits | None)."""
    from data.voxelize import voxelize_surface
    model_id, glb_path, mesh_dir, res = job
    cache = os.path.join(mesh_dir, model_id + ".npz")
    try:
        if os.path.exists(cache):
            m = np.load(cache)
            v, f = m["v"], m["f"]
        else:
            v, f = fetch_geometry(glb_path)
            np.savez_compressed(cache, v=v, f=f)
        vox = voxelize_surface(v, f, res)
        return model_id, None if vox is None else np.packbits(vox.reshape(-1))
    except Exception as e:
        print("fallo", model_id, repr(e)[:100], flush=True)
        return model_id, None


def main():
    from multiprocessing import Pool
    p = argparse.ArgumentParser()
    p.add_argument("--res", type=int, default=32)
    p.add_argument("--meshes", default=os.path.join(HERE, "raw"), help="carpeta de geometría (se descarga lo que falte)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    os.makedirs(args.meshes, exist_ok=True)
    with open(os.path.join(HERE, "categories.json"), encoding="utf-8") as f:
        types = json.load(f)
    meta = urllib.request.urlopen(BASE + "metadata/3dmodels.csv.gz", timeout=60).read()
    rows = sorted(csv.DictReader(io.TextIOWrapper(gzip.GzipFile(fileobj=io.BytesIO(meta)))),
                  key=lambda r: r["3dmodel_id"])[:args.limit]
    jobs = [(r["3dmodel_id"], r["path"], args.meshes, args.res) for r in rows]
    ids, bits = [], []
    with Pool(args.workers) as pool:
        for k, (model_id, b) in enumerate(pool.imap(_one, jobs, chunksize=8), 1):
            if b is not None:
                ids.append(model_id)
                bits.append(b)
            if k % 500 == 0:
                print(f"{k} / {len(jobs)}", flush=True)
    out = os.path.join(HERE, f"abo{args.res}.npz")
    np.savez_compressed(out, bits=np.stack(bits), ids=np.array(ids),
                        types=np.array([types.get(i, "UNKNOWN") for i in ids]), res=args.res)
    print(f"guardado {out}: {len(ids)} formas de {len(jobs)}")


if __name__ == "__main__":
    main()
