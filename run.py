#!/usr/bin/env python3
import argparse

from app import App
from config import Config, overrides


def main():
    p = argparse.ArgumentParser(description="cars - evolve a self-driving car with NEAT")
    p.add_argument("--pop", type=int, default=Config.pop_size)
    p.add_argument("--generations", type=int, default=Config.generations)
    p.add_argument("--scale", type=float, default=0.0)
    p.add_argument("--load", metavar="PATH",
                   help="carry on from a save (press s in the app to make one)")
    p.add_argument("--hard", action="store_true",
                   help="the hard track set: sharper, narrower, pillars on the hardest")
    p.add_argument("--set", nargs="*", default=[], metavar="FIELD=VALUE",
                   help="any Config field, e.g. --set n_rays=8 sharing=none")
    a = p.parse_args()
    hard = dict(track_hardness=1.6, track_obstacles=3) if a.hard else {}
    App(Config(pop_size=a.pop, generations=a.generations, ui_scale=a.scale, **hard, **overrides(a.set)), resume=a.load).run()


if __name__ == "__main__":
    main()
