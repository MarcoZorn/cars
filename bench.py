#!/usr/bin/env python3
"""Headless evolution benchmark: no window, several seeds in parallel.

Evolves a population for a number of generations on the usual track cycle,
then races the whole final population on every track and scores the best
generalist: how many of the tracks it laps, its mean lap fraction, and its
mean lap time in ticks over the tracks it does lap (once everything laps
every track, speed is what is left to tell runs apart). That same genome is
then scored on a held-out set of tracks it never raced in training. Judging
on all tracks, not on whichever track the last generation happened to race,
is what makes two runs comparable.

    python bench.py --seeds 8 --pop 200 --gens 60
"""
import argparse
import random
import time
from multiprocessing import Pool

import numpy as np

from car import Car, N_OUTPUTS, fitness, n_inputs, step_all
from config import Config
from neat_core.brain import Brain
from neat_core.population import Population
from tracks import build_tracks

HOLDOUT = 1000  # held-out tracks: same difficulty curve, different seeds


def race(genomes, track, cfg):
    cars = [Car(g, Brain(g), track, cfg) for g in genomes]
    for _ in range(cfg.max_ticks):
        if not any(c.alive for c in cars):
            break
        step_all(cars, track, cfg)
    return cars


def run(args):
    seed, pop_size, gens, overrides = args
    random.seed(seed)
    np.random.seed(seed)
    cfg = Config(pop_size=pop_size, **overrides)
    tracks = build_tracks(cfg)
    pop = Population(n_inputs(cfg), N_OUTPUTS, cfg)
    for _ in range(gens):
        track = tracks[pop.generation % len(tracks)]
        for c in race(pop.genomes, track, cfg):
            c.genome.fitness = fitness(c, track, cfg)
        pop.evolve()

    # every genome on every training track; the best generalist there is
    # then raced on tracks it has never seen
    best, train = pick(pop.genomes, tracks, cfg)
    held = Config(**{**cfg.__dict__, "track_seed": cfg.track_seed + HOLDOUT})
    _, test = pick([best], build_tracks(held), cfg)
    return train, test


def pick(genomes, tracks, cfg):
    """Race every genome on every track; return the best one and its laps,
    mean lap fraction and mean lap time (ticks, over the tracks it laps)."""
    laps = np.zeros(len(genomes))
    frac = np.zeros(len(genomes))
    ticks = np.zeros(len(genomes))
    for track in tracks:
        for i, c in enumerate(race(genomes, track, cfg)):
            laps[i] += c.finished
            frac[i] += min(1.0, c.best_progress / track.lap_length)
            ticks[i] += c.ticks if c.finished else 0
    b = int(np.argmax(laps * 10 + frac - ticks / (len(tracks) * cfg.max_ticks)))
    return genomes[b], (laps[b], frac[b] / len(tracks), ticks[b] / max(1, laps[b]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--seed0", type=int, default=1)
    p.add_argument("--pop", type=int, default=200)
    p.add_argument("--gens", type=int, default=60)
    p.add_argument("--set", nargs="*", default=[],
                   help="config overrides, e.g. p_weight=0.5 weight_step=0.15")
    a = p.parse_args()
    overrides = {}
    for kv in a.set:
        k, v = kv.split("=")
        kind = type(getattr(Config, k))
        # bool("False") is True, so booleans are read by name
        overrides[k] = v.lower() in ("1", "true", "yes") if kind is bool else kind(v)

    t0 = time.time()
    jobs = [(a.seed0 + s, a.pop, a.gens, overrides) for s in range(a.seeds)]
    with Pool(min(len(jobs), 30)) as pool:
        res = pool.map(run, jobs)
    print(f"{' '.join(a.set) or 'baseline'}  pop={a.pop} gens={a.gens} seeds={a.seeds}"
          f"  ({time.time() - t0:.0f}s)")
    for name, k in (("train", 0), ("held-out", 1)):
        laps = [r[k][0] for r in res]
        print(f"  {name:8} laps/{len(build_tracks(Config(**overrides)))} per seed:",
              " ".join(f"{x:.0f}" for x in laps))
        print(f"  {name:8} laps mean {np.mean(laps):.2f}"
              f"  lap-frac mean {np.mean([r[k][1] for r in res]):.3f}"
              f"  lap ticks mean {np.mean([r[k][2] for r in res]):.0f}")


if __name__ == "__main__":
    main()
