#!/usr/bin/env python3
import argparse

from app import App
from config import Config


def main():
    p = argparse.ArgumentParser(description="cars - evolve a self-driving car with NEAT")
    p.add_argument("--pop", type=int, default=Config.pop_size)
    p.add_argument("--generations", type=int, default=Config.generations)
    p.add_argument("--scale", type=float, default=0.0)
    p.add_argument("--hard", action="store_true",
                   help="the hard track set: sharper, narrower, pillars on the hardest")
    a = p.parse_args()
    hard = dict(track_hardness=1.6, track_obstacles=3) if a.hard else {}
    App(Config(pop_size=a.pop, generations=a.generations, ui_scale=a.scale, **hard)).run()


if __name__ == "__main__":
    main()
