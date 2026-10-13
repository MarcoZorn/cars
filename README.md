# cars

Self-driving cars evolved with NEAT. No hand-written driving logic - just
wall-detecting sensors, a neural net, and a genetic algorithm.

![600 cars driving an evolved neural network around a procedural track, with the leader's network drawn live](docs/demo.gif)

<sub>Generation 16 of a live run. The panel on the right is the leader's actual network:
its inputs are eight wall distances and its own speed, its outputs are steering and
throttle, and the hidden nodes between them were added by mutation, not by design.
A browser version you can run without installing anything is at
[neat-from-scratch](https://github.com/MarcoZorn/neat-from-scratch).</sub>

## What it does

- 1000 cars, 15 procedurally generated tracks, 250 generations
- each car has 8 raycast sensors in a full ring (front, back, sides, diagonals)
  plus its own speed as inputs, steering and throttle as outputs
- crashing doesn't end a car's race - it respawns at the start line and keeps
  trying until it finishes a lap or runs out of time
- fitness is the best single attempt, not the sum of attempts (so crash-spamming
  doesn't help)
- standard NEAT: speciation, fitness sharing, elitism, crossover, mutation

Tracks cycle each generation so the population doesn't just memorize one loop.

## Running it

```
pip install -r requirements.txt
python run.py
```

Controls:

- `space` pause
- `f` speed (1x / 4x / 16x)
- `+` / `-` / mouse wheel zoom, drag to pan, `0` to fit
- `c` follow the leader car
- `[` `]` UI text size
- `q` quit

Headless, no window - evolve several seeds in parallel and score each run's
best generalist on all 15 tracks (laps, lap fraction, lap time):

```
python bench.py --seeds 8 --pop 200 --gens 20
python bench.py --set p_weight=0.5 weight_step=0.15   # any Config field
python bench.py --set track_hardness=1.6 track_obstacles=3   # the hard set
```

The winner is also raced on fifteen held-out tracks it never saw in
training, same difficulty curve, different seeds. On the standard set the
task saturates (by generation 40 nearly every run laps every training
track), and the held-out score is what still separates runs. The hard set
(sharper, narrower tracks, pillars on the hardest) does not saturate.
16 seeds, population 200, 40 generations:

| | training laps / 15 | held-out laps / 15 | gap |
|---|---|---|---|
| standard tracks | 14.4 | 13.1 | 1.3 |
| hard tracks | 12.9 | 10.8 | 2.1 |

Tried on the hard set and not kept: scoring progress as a share of each
track's lap instead of in pixels (the tracks differ in length by about 2x, so
raw distance is not comparable from one generation to the next) came out at
13.0 and 11.0, inside the noise between seeds.

## Files

- `neat_core/` - genome, mutation, crossover, speciation (standard NEAT)
- `tracks.py` - track generation (polar functions, always simple closed loops)
- `car.py` - physics, sensors, fitness
- `app.py` / `render.py` - the live viewer
- `bench.py` - headless multi-seed benchmark
- `test_cars.py` - `python test_cars.py`

MIT licensed.
