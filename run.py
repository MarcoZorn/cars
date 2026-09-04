#!/usr/bin/env python3
import argparse

from app import App
from config import Config


def main():
    p = argparse.ArgumentParser(description="cars - evolve a self-driving car with NEAT")
    p.add_argument("--pop", type=int, default=Config.pop_size)
    p.add_argument("--generations", type=int, default=Config.generations)
    p.add_argument("--scale", type=float, default=0.0)
    a = p.parse_args()
    App(Config(pop_size=a.pop, generations=a.generations, ui_scale=a.scale)).run()


if __name__ == "__main__":
    main()
