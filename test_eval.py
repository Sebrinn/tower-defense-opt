"""Test ręczny evaluate: jedna wieża single przy fali 'fast' (ścieżka 0)."""
import numpy as np
from maps import default_instance
from eval import new_layout, evaluate

inst = default_instance(budget=100)
x = new_layout(inst)
c = inst.cells.index((5, 4))                    # tuż obok ścieżki 0 (y=3)
x[c, 0] = 1                                     # wieża 'single' (dps10, zasięg 3)
n_cov = int(inst.cov[c, 0, 0].sum())            # liczba pól ścieżki 0 w zasięgu
expected = 10 * 1.0 * 0.5 * n_cov               # dps * m * tau(0.5s) * pola
leak0 = evaluate(inst, x)[1][0]
print("pola w zasięgu:", n_cov, "| oczekiwane zdjęte HP:", expected, "| przeciek fali 0:", leak0)
assert abs((inst.H[0] - leak0) - expected) < 1e-9

c2 = inst.cells.index((5, 2)); x[c2, 3] = 1     # + spowalniacz po drugiej stronie
leak1 = evaluate(inst, x)[1][0]
print("ze spowalniaczem przeciek:", leak1)
assert leak1 < leak0
print("OK")
