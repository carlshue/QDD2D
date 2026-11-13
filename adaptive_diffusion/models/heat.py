"""
Medición pasiva de nodos "calientes" y "fríos" durante el muestreo.

No cambia las muestras: sólo anota, en cada evaluación de la red, cuántos nodos
siguen en juego, para estimar cuánto cómputo se ahorraría si sólo se volvieran a
mirar las zonas calientes (p. ej. con atención por ventanas).

  caliente  nodo sin decidir (MASK) o cuyo estado actual recibe de la red una
            probabilidad menor que `thr` (la red "quiere" cambiarlo)
  ventana   todos los nodos que cuelgan de una misma celda del nivel
            `window_level` (los niveles más gruesos forman una ventana aparte);
            una ventana es caliente si contiene algún nodo caliente
"""
import torch


class HeatTrace:
    def __init__(self, pyr, thr=0.9, window_levels=(2, 3)):
        self.pyr, self.thr, self.window_levels = pyr, thr, tuple(window_levels)
        self.win = {}
        for wl in self.window_levels:
            sh = (pyr.level - wl).clamp_min(0)
            wid = 1 + (pyr.y >> sh) * (2 ** wl) + (pyr.x >> sh)
            self.win[wl] = torch.where(pyr.level >= wl, wid, torch.zeros_like(wid))
        self.steps = []

    def record(self, ex, p_current, changed, tag=""):
        """ex: nodos presentes al evaluar la red (num, size). p_current: prob. que
        la red da al estado actual (0 en nodos MASK). changed: nodos presentes
        cuyo estado cambia tras esta evaluación."""
        num = ex.shape[0]
        hot = ex & (p_current < self.thr)
        changed = changed & ex
        row = {"fase": tag, "nodos": ex.sum().item() / num, "calientes": hot.sum().item() / num,
               "cambian": changed.sum().item() / num, "cambian_frios": (changed & ~hot).sum().item() / num}
        for wl, wid in self.win.items():
            n_win = 1 + 4 ** wl
            idx = wid[None].expand(num, -1)
            hot_w = torch.zeros(num, n_win, device=ex.device).scatter_add_(1, idx, hot.float()) > 0
            any_w = torch.zeros(num, n_win, device=ex.device).scatter_add_(1, idx, ex.float()) > 0
            in_hot = ex & hot_w.gather(1, idx)
            row[f"tokens_ventana_caliente_n{wl}"] = in_hot.sum().item() / num
            row[f"ventanas_n{wl}"] = any_w.sum().item() / num
            row[f"ventanas_calientes_n{wl}"] = (hot_w & any_w).sum().item() / num
            row[f"cambian_fuera_n{wl}"] = (changed & ~in_hot).sum().item() / num
        self.steps.append(row)

    def summary(self):
        tot = {k: sum(r[k] for r in self.steps) for k in self.steps[0] if k != "fase"}
        out = {"evaluaciones": len(self.steps), "tokens_por_muestra": round(tot["nodos"], 1),
               "umbral": self.thr,
               "nodos_calientes_pct": round(100 * tot["calientes"] / tot["nodos"], 1),
               "nodos_que_cambian_pct": round(100 * tot["cambian"] / tot["nodos"], 1),
               "cambios_en_nodos_frios_pct": round(100 * tot["cambian_frios"] / max(tot["cambian"], 1e-9), 1)}
        for wl in self.window_levels:
            out[f"ventanas_n{wl}"] = {
                "tokens_por_ventana": round(tot["nodos"] / tot[f"ventanas_n{wl}"], 1),
                "ventanas_calientes_pct": round(100 * tot[f"ventanas_calientes_n{wl}"] / tot[f"ventanas_n{wl}"], 1),
                "tokens_a_recalcular_pct": round(100 * tot[f"tokens_ventana_caliente_n{wl}"] / tot["nodos"], 1),
                "cambios_fuera_de_ventanas_calientes_pct":
                    round(100 * tot[f"cambian_fuera_n{wl}"] / max(tot["cambian"], 1e-9), 1)}
        return out
