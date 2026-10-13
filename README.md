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
python run.py --hard   # sharper, narrower tracks, pillars on the hardest
python run.py --load saves/gen0040.pkl.gz   # carry on from a save
```

Controls:

- `space` pause
- `f` speed (1x / 4x / 16x)
- `+` / `-` / mouse wheel zoom, drag to pan, `0` to fit
- `c` follow the leader car
- `s` save the population to `saves/` (resume it with `--load`)
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
16 seeds, population 200, 40 generations, with the defaults as they are now
(weight step 0.6 and fitness sharing fixed, both below), and in brackets as
they were before either:

| | training laps / 15 | held-out laps / 15 | gap |
|---|---|---|---|
| standard tracks | 14.9 (14.4) | 13.7 (13.1) | 1.2 (1.3) |
| hard tracks | 14.1 (12.9) | 12.9 (10.8) | 1.2 (2.1) |

Bigger weight mutations generalise better. With the weight step raised from
0.35 to 0.6 (now the default), over 32 seeds on the hard set the held-out
score went from 11.0 to 12.3 laps, and on the standard set from 13.1 to 13.5,
with training scores up as well. A wider search finds drivers that rely less
on the exact shape of the tracks they trained on.

Tried on the hard set and not kept: scoring progress as a share of each
track's lap instead of in pixels (the tracks differ in length by about 2x, so
raw distance is not comparable from one generation to the next) came out at
13.0 and 11.0, inside the noise between seeds.

Do the rear rays earn their place? Five rays over the front half (the same
five angles as the front of the ring: ahead, the diagonals, and both sides)
against the full ring of eight, 32 seeds on the hard set:

| | training laps / 15 | held-out laps / 15 | held-out lap time, ticks |
|---|---|---|---|
| ring of 8 | 13.5 | 12.1 | 248 |
| front 5 | 13.8 | 12.1 | 274 |
| front 5 + back diagonals | 13.8 | 11.9 | 281 |

Exactly as many laps either way, but a driver with the whole ring laps about
a tenth faster. My guess was the back diagonals: coming out of a corner they
still see the wall the car is pulling away from. Wrong: with them and
without the ray pointing straight back (`n_rays=7 fov=1.5pi`) it is as slow
as the front five alone. Whatever the full ring gives, it takes the ray
that looks straight back, or the whole set. The ring stays.

**A bug in fitness sharing.** NEAT divides each genome's fitness by the size
of its species, and gives each species offspring in proportion to the sum of
those shares, which comes to the species' mean fitness. This code averaged
the shares instead of summing them, dividing by the species' size a second
time, so a species twice as big got half the offspring it should have. With
the standard rule (`sharing="mean"`, now the default), hard set, two batches
of 32 seeds:

| offspring per species | training laps / 15 | held-out laps / 15 | held-out lap time |
|---|---|---|---|
| mean fitness / size (the bug) | 13.5 / 13.2 | 12.1 / 11.8 | 248 / 232 |
| mean fitness (standard) | 13.9 / 13.8 | 12.5 / 12.4 | 260 / 243 |
| total fitness (no sharing) | 14.1 | 12.5 | 279 |

Half a lap more on unseen tracks in both batches, at laps about 5% slower.
No sharing at all matches it on laps and is slower still. With the fix in, the
weight step still holds up (hard set, 32 seeds, held-out laps): 11.5 at
0.35, 12.5 at 0.6, 12.6 at 0.9. It stays at 0.6. The comparisons
above this one, other than the first table, were measured with the bug in
place.

**Does a car need to feel its own speed?** The network has no memory, so
without the speed input it cannot tell how fast it is going; the rays look
the same at any speed. I expected that to be crippling. With the input
silenced (`speed_input=False`), hard set, the same 32 seeds:

| | training laps / 15 | held-out laps / 15 | held-out lap time |
|---|---|---|---|
| with speed | 13.9 | 12.5 | 260 |
| without | 13.6 | 12.3 | 236 |

A quarter of a lap worse, and faster laps. Not knowing your speed seems to
mean keeping your foot down and steering by the walls alone, which crashes
a little more and laps a tenth quicker when it works. The input stays, but
it is doing much less than I thought.

## Files

- `neat_core/` - genome, mutation, crossover, speciation (standard NEAT)
- `tracks.py` - track generation (polar functions, always simple closed loops)
- `car.py` - physics, sensors, fitness
- `app.py` / `render.py` - the live viewer
- `bench.py` - headless multi-seed benchmark
- `test_cars.py` - `python test_cars.py`

MIT licensed.
