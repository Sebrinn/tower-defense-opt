"""Instancja problemu: mapa, ścieżki, typy wież/wrogów, fale + macierze cov i A."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class TowerType:
    name: str
    cost: float
    rng: float            # zasięg w polach
    dps: float            # obrażenia/s (0 dla spowalniacza)
    mult: dict = field(default_factory=dict)  # m_{k,e}
    aoe_hits: float = 1.0  # ilu wrogów z roju trafia naraz
    slow: bool = False


@dataclass
class EnemyType:
    name: str
    hp: float
    speed: float          # pola/s
    swarm: bool = False


@dataclass
class Wave:
    path: int             # indeks ścieżki
    enemy: str
    n: int
    weight: float = 1.0


def expand_path(waypoints):
    """Łamana o osiowych odcinkach -> lista pól (x, y)."""
    cells = [waypoints[0]]
    for (x0, y0), (x1, y1) in zip(waypoints, waypoints[1:]):
        dx, dy = np.sign(x1 - x0), np.sign(y1 - y0)
        x, y = x0, y0
        while (x, y) != (x1, y1):
            x, y = x + dx, y + dy
            cells.append((x, y))
    return cells


class Instance:
    def __init__(self, size, paths, towers, enemies, waves, budget, sigma=0.5):
        self.size, self.budget, self.sigma = size, budget, sigma
        self.paths = [expand_path(p) for p in paths]
        self.towers = towers
        self.enemies = {e.name: e for e in enemies}
        self.waves = waves
        on_path = {c for p in self.paths for c in p}
        self.cells = [(x, y) for x in range(size) for y in range(size)
                      if (x, y) not in on_path]            # zbiór C
        self.K = len(towers)
        self.F = len(waves)
        self.P = max(len(self.paths[w.path]) for w in waves)
        self.cost = np.array([t.cost for t in towers])
        self.is_slow = np.array([t.slow for t in towers])
        self.H = np.array([w.n * self.enemies[w.enemy].hp for w in waves])
        self.w = np.array([w.weight for w in waves])
        self._build()

    def _build(self):
        C, K, F, P = len(self.cells), self.K, self.F, self.P
        pos = np.array(self.cells, float)
        self.cov = np.zeros((C, K, F, P), bool)
        self.A = np.zeros((C, K, F, P))
        self.tau = np.zeros((F, P))
        for f, wv in enumerate(self.waves):
            en = self.enemies[wv.enemy]
            path = np.array(self.paths[wv.path], float)
            self.tau[f, :len(path)] = 1.0 / en.speed       # czas na jednym polu
            d = np.linalg.norm(pos[:, None, :] - path[None, :, :], axis=2)
            for k, t in enumerate(self.towers):
                c = d <= t.rng
                self.cov[:, k, f, :len(path)] = c
                if not t.slow:
                    hits = t.aoe_hits if en.swarm else 1.0
                    m = t.mult.get(wv.enemy, 1.0)
                    self.A[:, k, f, :len(path)] = (
                        c * t.dps * m * hits * self.tau[f, :len(path)])


def default_instance(budget=100, sigma=0.5):
    towers = [
        TowerType("single", 10, 3.0, 10, {"fast": 1.0, "armored": 0.4, "swarm": 1.0}),
        TowerType("aoe", 15, 2.5, 6, {"fast": 0.8, "armored": 0.3, "swarm": 1.0}, aoe_hits=4),
        TowerType("antiarmor", 14, 3.0, 8, {"fast": 0.6, "armored": 1.5, "swarm": 0.5}),
        TowerType("slow", 8, 3.0, 0, slow=True),
    ]
    enemies = [EnemyType("fast", 30, 2.0), EnemyType("armored", 200, 0.7),
               EnemyType("swarm", 15, 1.0, swarm=True)]
    paths = [[(0, 3), (10, 3), (10, 12), (19, 12)],
             [(0, 17), (6, 17), (6, 8), (14, 8), (14, 19)],
             [(19, 0), (19, 5), (12, 5), (12, 12)]]
    waves = [Wave(0, "fast", 20), Wave(1, "armored", 8), Wave(2, "swarm", 40)]
    return Instance(20, paths, towers, enemies, waves, budget, sigma)
