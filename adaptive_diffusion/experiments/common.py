"""Utilidades compartidas: elegir dispositivo, cargar un modelo y generar con él."""
import numpy as np
import torch

from models.quadtree_transformer import QuadtreeTransformer


def pick_device(name="auto"):
    if name != "auto":
        if name.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit("Se pidió CUDA pero PyTorch no detecta ninguna GPU.")
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(path, device):
    ck = torch.load(path, map_location=device)
    a = ck["args"]
    method = a.get("method", "grow")
    model = QuadtreeTransformer(depth=ck["depth"], dim=a["dim"], layers=a["layers"], heads=a["heads"],
                                use_time=(method in ("edit", "grow_merge"))).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, {"method": method, "depth": ck["depth"], "prior": ck.get("prior"), "args": a,
                   "dataset": a.get("dataset", "mnist")}


@torch.no_grad()
def generate(model, meta, labels, device, steps=None, temperature=1.0, batch=250, resolve_fresh=False,
             revise=1.0, revise_thr=0.5, trace=None, remask=0.0, remask_rounds=0, remask_mode="conf"):
    """Genera una imagen por etiqueta. Devuelve (imgs uint8 (N,S,S), estadísticas medias).
    resolve_fresh sólo afecta a 'edit'; revise y revise_thr, a 'grow_merge'; remask*, a 'grow'.
    trace: models.heat.HeatTrace opcional (sólo 'edit' y 'grow_merge')."""
    use_amp = str(device).startswith("cuda")
    imgs, agg = [], {}
    for lab in labels.split(batch):
        lab = lab.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            if meta["method"] == "edit":
                from models.edit_diffusion import sample
                from structures.pyramid import Pyramid
                pyr = Pyramid(meta["depth"], device)
                out, st = sample(model, pyr, meta["prior"].to(device), len(lab), steps or 32, lab, temperature,
                                 resolve_fresh=resolve_fresh, trace=trace)
                out = out.cpu().numpy().astype(np.uint8)
            elif meta["method"] == "grow_merge":
                from models.grow_merge import sample
                from structures.pyramid import Pyramid
                pyr = Pyramid(meta["depth"], device)
                out, st = sample(model, pyr, len(lab), steps or 8, lab, temperature, revise=revise,
                                 revise_thr=revise_thr, trace=trace)
                out = out.cpu().numpy().astype(np.uint8)
            else:
                from models.adaptive_diffusion import sample
                from structures.binary_quadtree import levels_to_image, tokens_to_levels
                rs = {}
                trees, evals, tok = sample(model, len(lab), steps or 8, lab, temperature, device, remask=remask,
                                           remask_rounds=remask_rounds, remask_mode=remask_mode, stats=rs)
                out = np.stack([levels_to_image([lv[None] for lv in tokens_to_levels(t.numpy(), meta["depth"])])[0]
                                for t in trees])
                sizes = [len(t) for t in trees]
                st = {"evals": evals, "tokens_por_muestra": tok, "nodos_final": float(np.mean(sizes)),
                      "nodos_max": int(max(sizes)), **rs}
        imgs.append(out)
        for k, v in st.items():
            agg.setdefault(k, []).append(v)
    stats = {k: (max(v) if k == "nodos_max" else float(np.mean(v))) for k, v in agg.items()}
    return np.concatenate(imgs), stats
