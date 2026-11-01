import time
import matplotlib.pyplot as plt
from maps import default_instance
from eval import evaluate, new_layout, is_feasible
from baselines import *
from plot import plot_layout, plot_comparison

inst = default_instance(budget=100)
print(f"pola: {len(inst.cells)}, wieże: {inst.K}, fale: {inst.F}, H={inst.H}, budżet={inst.budget}")
print("brak wież:", round(evaluate(inst, new_layout(inst))[0], 1))

res, layouts = {}, {}
t = time.time(); x, mean = baseline_random(inst); res["losowy (średnia)"] = mean
layouts["losowy (najlepszy)"] = x
for name, fn in [("najbliżej ścieżki", baseline_nearest),
                 ("greedy w próżni", baseline_greedy_vacuum),
                 ("greedy krańcowy", greedy_marginal)]:
    t = time.time(); x = fn(inst); dt = time.time() - t
    assert is_feasible(inst, x)
    res[name] = evaluate(inst, x)[0]; layouts[name] = x
    print(f"{name:20s} przeciek={res[name]:8.1f}  czas={dt:.2f}s  per fala={evaluate(inst, x)[1].round(0)}")
print(f"{'losowy (średnia)':20s} przeciek={mean:8.1f}")
plot_comparison(res)
fig, axs = plt.subplots(1, 2, figsize=(11, 5.5))
plot_layout(inst, layouts["greedy w próżni"], axs[0], "greedy w próżni")
plot_layout(inst, layouts["greedy krańcowy"], axs[1], "greedy krańcowy")
fig.savefig("layouts.png", dpi=120)
