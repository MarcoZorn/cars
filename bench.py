#!/usr/bin/env python3
"""Headless evolution benchmark: no window, several seeds in parallel.

Evolves a population for a number of generations on the usual track cycle,
then races the whole final population on every track and scores the best
generalist: how many of the tracks it laps, and its mean lap fraction. Judging
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

    # every genome on every track: laps and lap fraction per genome
    laps = np.zeros(len(pop.genomes))
    frac = np.zeros(len(pop.genomes))
    for track in tracks:
        for i, c in enumerate(race(pop.genomes, track, cfg)):
            laps[i] += c.finished
            frac[i] += min(1.0, c.best_progress / track.lap_length)
    best = int(np.argmax(laps * 10 + frac))
    return laps[best], frac[best] / len(tracks)


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
        overrides[k] = type(getattr(Config, k))(v)

    t0 = time.time()
    jobs = [(a.seed0 + s, a.pop, a.gens, overrides) for s in range(a.seeds)]
    with Pool(min(len(jobs), 30)) as pool:
        res = pool.map(run, jobs)
    laps = [r[0] for r in res]
    fracs = [r[1] for r in res]
    print(f"{' '.join(a.set) or 'baseline'}  pop={a.pop} gens={a.gens} seeds={a.seeds}")
    print("  laps/15 per seed:", " ".join(f"{x:.0f}" for x in laps))
    print(f"  laps mean {np.mean(laps):.2f}  lap-frac mean {np.mean(fracs):.3f}"
          f"  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
