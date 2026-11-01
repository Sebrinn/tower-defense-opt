import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = ["tab:blue", "tab:green", "tab:red", "tab:purple", "tab:orange"]


def plot_layout(inst, x, ax=None, title=""):
    ax = ax or plt.subplots(figsize=(6, 6))[1]
    ax.set_xlim(-.5, inst.size - .5); ax.set_ylim(-.5, inst.size - .5)
    ax.set_aspect("equal"); ax.set_title(title)
    for i, p in enumerate(inst.paths):
        xs, ys = zip(*p)
        ax.plot(xs, ys, "-", lw=6, color="lightgray", zorder=1)
        ax.annotate(f"wejście {i}", p[0], fontsize=7)
    for c, k in zip(*np.nonzero(x)):
        cx, cy = inst.cells[c]
        ax.scatter(cx, cy, s=70, color=COL[k], zorder=3,
                   marker="s" if inst.is_slow[k] else "o")
    for k, t in enumerate(inst.towers):
        ax.scatter([], [], color=COL[k], label=t.name)
    ax.legend(fontsize=7, loc="upper right")
    return ax


def plot_comparison(results, fname="baselines.png"):
    names, vals = zip(*results.items())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, vals, color="steelblue")
    ax.set_ylabel("ważony przeciek HP (mniej = lepiej)")
    plt.xticks(rotation=20); plt.tight_layout(); fig.savefig(fname, dpi=130)
