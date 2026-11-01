import numpy as np
from eval import new_layout, evaluate


def _spent(inst, x):
    return float((x @ inst.cost).sum())


def random_layout(inst, rng):
    x = new_layout(inst)
    for _ in range(2000):
        c, k = rng.integers(len(inst.cells)), rng.integers(inst.K)
        if x[c].sum() == 0 and _spent(inst, x) + inst.cost[k] <= inst.budget:
            x[c, k] = 1
    return x


def baseline_random(inst, n=100, seed=0):
    rng = np.random.default_rng(seed)
    runs = [random_layout(inst, rng) for _ in range(n)]
    scores = [evaluate(inst, x)[0] for x in runs]
    return runs[int(np.argmin(scores))], float(np.mean(scores))   # (najlepszy układ, średnia)


def baseline_nearest(inst):
    """Najtańsze wieże obrażeń na polach najbliżej dowolnej ścieżki."""
    pos = np.array(inst.cells, float)
    allp = np.array([p for path in inst.paths for p in path], float)
    d = np.linalg.norm(pos[:, None] - allp[None], axis=2).min(axis=1)
    k = min((k for k in range(inst.K) if not inst.is_slow[k]), key=lambda k: inst.cost[k])
    x = new_layout(inst)
    for c in np.argsort(d):
        if _spent(inst, x) + inst.cost[k] > inst.budget:
            break
        x[c, k] = 1
    return x


def baseline_greedy_vacuum(inst):
    """Greedy po wartości/koszt liczonej 'w próżni' (bez nasycenia i synergii)."""
    val = (inst.A * inst.w[None, None, :, None]).sum(axis=(2, 3))     # (C,K)
    ratio = val / inst.cost[None, :]
    x = new_layout(inst)
    for idx in np.argsort(-ratio, axis=None):
        c, k = divmod(int(idx), inst.K)
        if ratio[c, k] <= 0:
            break
        if x[c].sum() == 0 and _spent(inst, x) + inst.cost[k] <= inst.budget:
            x[c, k] = 1
    return x


def greedy_marginal(inst):
    """Greedy z zyskiem krańcowym liczonym przez evaluate (opcjonalny baseline 4)."""
    x = new_layout(inst)
    cur = evaluate(inst, x)[0]
    while True:
        best, best_gain = None, 1e-9
        left = inst.budget - _spent(inst, x)
        for c in np.where(x.sum(axis=1) == 0)[0]:
            for k in range(inst.K):
                if inst.cost[k] > left:
                    continue
                x[c, k] = 1
                gain = (cur - evaluate(inst, x)[0]) / inst.cost[k]
                x[c, k] = 0
                if gain > best_gain:
                    best, best_gain = (c, k), gain
        if best is None:
            return x
        x[best] = 1
        cur = evaluate(inst, x)[0]
