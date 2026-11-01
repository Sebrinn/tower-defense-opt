"""Funkcja oceny układu wież -- wspólna dla wszystkich metod (i MILP do testów)."""
import numpy as np


def new_layout(inst):
    return np.zeros((len(inst.cells), inst.K), dtype=int)   # x_{c,k}


def is_feasible(inst, x):
    return x.sum(axis=1).max() <= 1 and (x @ inst.cost).sum() <= inst.budget + 1e-9


def evaluate(inst, x):
    """Zwraca (ważony przeciek, przeciek per fala). Mniej = lepiej."""
    slow_x = x * inst.is_slow                                   # (C,K)
    u = np.einsum("ck,ckfp->fp", slow_x, inst.cov) > 0          # odcinek spowolniony
    dmg_x = x * ~inst.is_slow
    raw = np.einsum("ck,ckfp->fp", dmg_x, inst.A)               # obrażenia bez spowolnienia
    z = raw * (1 + inst.sigma * u)                              # z_{f,p}
    killed = np.minimum(z.sum(axis=1), inst.H)                  # nasycenie: sum_p z <= H_f
    leak = inst.H - killed
    return float((inst.w * leak).sum()), leak
