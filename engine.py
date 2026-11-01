"""Evidence-indexed occupancy inference and action-conditioned route forecasts.

This intentionally exposes the additive sufficient-statistic assumption. Retraction
is exact bookkeeping, NOT a learned inverse of an arbitrary recurrent model.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def features(raw, quality):
    raw, quality = np.broadcast_arrays(raw, quality)
    return np.stack([raw, raw * quality, quality, np.ones_like(raw)], axis=-1)


def fit_sensor(seed=20261101, n=12000, steps=180):
    """Tiny logistic observation model, trained on synthetic detector features."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    quality = rng.uniform(.25, 1, n)
    raw = (2 * y - 1) * 1.15 + rng.normal(0, .35 + 1.2 * (1 - quality))
    x = features(raw, quality)
    w = np.zeros(4)
    losses = []
    for i in range(steps):
        p = sigmoid(x @ w)
        w -= .22 * (x.T @ (p - y) / n + .001 * w)
        if i % 10 == 0:
            losses.append(float(-np.mean(y * np.log(p + 1e-10) + (1-y) * np.log(1-p+1e-10))))
    y2 = rng.integers(0, 2, 4000)
    q2 = rng.uniform(.25, 1, 4000)
    r2 = (2*y2-1)*1.15 + rng.normal(0, .35+1.2*(1-q2))
    p2 = sigmoid(features(r2, q2) @ w)
    return w, {"seed": seed, "train_examples": n, "test_examples": 4000,
               "steps": steps, "loss_trace": losses,
               "test_accuracy": float(np.mean((p2 >= .5) == y2)),
               "test_brier": float(np.mean((p2-y2)**2)),
               "description": "synthetic binary sensor classifier; balanced class prior"}


@dataclass(frozen=True)
class Evidence:
    id: str
    cell: int
    delta: float
    source: str = "camera"
    timestamp: int = 0


class State:
    def __init__(self, prior):
        self.prior = np.asarray(prior, dtype=float).copy()
        self.logits = self.prior.copy()
        self.ledger = {}
        self.active = set()
        self.cache = {}
        self.dependencies = {}

    def add(self, evidence):
        if evidence.id in self.ledger:
            raise ValueError("duplicate evidence identifier")
        if not 0 <= evidence.cell < len(self.logits) or not math.isfinite(evidence.delta):
            raise ValueError("invalid evidence")
        self.ledger[evidence.id] = evidence
        self.active.add(evidence.id)
        self.logits[evidence.cell] += evidence.delta
        self.invalidate({evidence.cell})

    def invalidate(self, cells):
        affected = set().union(*(self.dependencies.get(c, set()) for c in cells))
        for key in affected:
            self.cache.pop(key, None)
        return affected

    def retract(self, identifiers):
        """Unknown IDs fail atomically; already retracted IDs are idempotent."""
        if any(i not in self.ledger for i in identifiers):
            raise KeyError("unknown evidence identifier")
        cells = set()
        for i in set(identifiers):
            if i in self.active:
                e = self.ledger[i]
                self.logits[e.cell] -= e.delta
                self.active.remove(i)
                cells.add(e.cell)
        return self.invalidate(cells)

    def forecast(self, key, path):
        """A route forecast depends on its cells, not only its evidence packet.

        New candidate selection can depend on all map cells; planning is therefore
        rerun globally after revision, rather than invalidated by this local rule.
        """
        path = tuple(path)
        old = self.cache.get(key)
        if old and old[0] == path:
            return old[1]
        # Static occupancy is a property of a cell, not a fresh draw per visit.
        p = sigmoid(self.logits[list(dict.fromkeys(path))])
        value = {"survival": float(np.exp(np.log1p(-np.minimum(p, 1-1e-12)).sum())),
                 "risk_cost": float((-np.log1p(-np.minimum(p, 1-1e-12))).sum())}
        self.cache[key] = (path, value)
        for cell in path:
            self.dependencies.setdefault(cell, set()).add(key)
        return value


def replay(prior, evidence, removed=()):
    state = np.asarray(prior, dtype=float).copy()
    removed = set(removed)
    selected = [e for e in evidence if e.id not in removed]
    if selected:
        np.add.at(state, [e.cell for e in selected], [e.delta for e in selected])
    return state


def plan(logits, width, height, start, goal):
    """Risk-aware A*, with conservative blockage threshold .55."""
    p = sigmoid(np.asarray(logits))
    queue = [(0., start)]
    cost, previous = {start: 0.}, {}
    while queue:
        _, cell = heapq.heappop(queue)
        if cell == goal:
            path = [cell]
            while cell in previous:
                cell = previous[cell]
                path.append(cell)
            return path[::-1]
        x, y = cell % width, cell // width
        for nx, ny in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            nxt = ny*width+nx
            if p[nxt] >= .55 and nxt != goal:
                continue
            proposed = cost[cell] + 1 + 3 * -math.log(max(1-p[nxt], 1e-12))
            if proposed < cost.get(nxt, float('inf')):
                cost[nxt], previous[nxt] = proposed, cell
                h = abs(nx-goal%width)+abs(ny-goal//width)
                heapq.heappush(queue, (proposed+h, nxt))
    return []


def scene(w):
    width, height = 28, 18
    truth = np.zeros(width*height, dtype=int)
    prior = np.full(width*height, math.log(.025/.975))
    for y in range(height):
        for x in range(width):
            wall = x in (0, width-1) or y in (0, height-1) or (x == 14 and y not in (4, 12, 13, 14))
            if wall:
                truth[y*width+x] = 1
                prior[y*width+x] = math.log(.995/.005)
    bad_cell = 4*width+14
    truth[bad_cell] = 1
    prior[bad_cell] = math.log(.78/.22)
    events = []
    for i, (x,y,r,q) in enumerate([(14,13,-1.8,.95),(14,12,-1.6,.9),(14,14,-1.5,.9),
                                  (7,8,-1.4,.9),(20,9,-1.5,.9),(14,4,-2.2,.98)]):
        events.append({"id": f"E{i+1:03}", "cell": y*width+x, "raw": r, "quality": q,
                       "delta": float(np.clip(features(r,q) @ w, -8, 8)),
                       "source": "camera-B" if i == 5 else "camera-A", "timestamp": i+1,
                       "valid": i != 5})
    return {"width": width,"height":height,"truth":truth.tolist(),"prior":prior.tolist(),
            "events":events,"bad_ids":["E006"],"start":4*width+3,"goal":4*width+24,
            "bad_cell":bad_cell,"title":"A passage that was never open"}
