"""A car: what it is, and the physics/sensing that run the whole population at
once.

Arcade physics, not a real bicycle model - throttle changes speed, steering
changes heading, and steering matters less the slower you are going, the same
way a stationary car does not turn by twisting the wheel. The car only ever
knows what its sensors tell it: a fan of rays reporting distance to the
nearest wall, plus its own speed. Nothing about the track shape or its
position is handed to the brain directly.

Sensing and physics run on the whole population as numpy arrays in one pass -
a thousand cars times a python-level ray march would be tens of millions of
tiny calls a tick. The one thing that cannot be batched is the brain itself:
every genome can have a different network topology, so evaluating a brain
stays a small per-car python call - but the networks here are a handful of
nodes each, so that cost is negligible next to the geometry.
"""
import math

import numpy as np

N_OUTPUTS = 2  # steering, throttle


def n_inputs(cfg):
    return cfg.n_rays + 1  # rays + own speed


def ray_angles(cfg):
    """Angle offsets for each sensor ray, relative to heading.

    An open cone (fov < a full circle) should hit both of its extreme edges,
    so the endpoints are included. A full 360-degree ring has no edges - the
    last ray is the same direction as the first would be - so the endpoint is
    dropped there instead, or one direction would be sensed twice and another
    (the true opposite of ray 0) would be missing.

    Shared by the physics and the renderer so what a car senses and what is
    drawn as its sensor fan can never silently drift apart.
    """
    half = cfg.fov / 2
    if cfg.n_rays <= 1:
        return [0.0]
    if abs(cfg.fov - 2 * math.pi) < 1e-9:
        return [-half + cfg.fov * i / cfg.n_rays for i in range(cfg.n_rays)]
    return [-half + cfg.fov * i / (cfg.n_rays - 1) for i in range(cfg.n_rays)]


class Car:
    """A car's race, not just its current attempt.

    A crash does not end the race - it ends the attempt. The car respawns at
    the start line and tries again with the same brain, until it either
    completes a lap or the generation's shared time budget runs out. Fitness
    is judged on the best single attempt, not the sum of them, so crashing on
    purpose to rack up cheap extra distance buys nothing: only how far one
    continuous attempt got, ever, counts.
    """
    __slots__ = ("genome", "brain", "x", "y", "a", "v", "alive", "ticks",
                 "stall", "theta0", "theta_unwrapped", "last_theta",
                 "progress", "best_progress", "lives", "finished",
                 "sensors", "out")

    def __init__(self, genome, brain, track, cfg):
        self.genome = genome
        self.brain = brain
        self.alive = True
        self.ticks = 0
        self.best_progress = 0.0
        self.lives = 1
        self.finished = False
        self.sensors = [0.0] * n_inputs(cfg)
        self.out = [0.0, 0.0]
        self.respawn(track)

    def respawn(self, track):
        """Reset kinematics and this attempt's progress. `ticks` (the shared
        time budget) and `best_progress` (the record of the race so far) are
        not touched - only the attempt itself restarts."""
        self.x, self.y = track.start_xy
        self.a = track.start_heading
        self.v = 0.0
        self.stall = 0
        theta0, _ = track.from_world(self.x, self.y)
        self.theta0 = float(theta0)
        self.theta_unwrapped = 0.0
        self.last_theta = float(theta0)
        self.progress = 0.0


def sense_all(cars, track, cfg):
    """Ray march every alive car's whole sensor fan in one batch.

    For each car and each ray, sample points at increasing distance and find
    the first one that has left the track. Vectorised the same way as a
    nearest-neighbour query: build every (car, ray, sample) point at once,
    test them all against the track in a single call, then pick out the
    first miss per (car, ray) with numpy instead of a python loop.
    """
    n = len(cars)
    steps = 20
    t = np.linspace(0.0, 1.0, steps)[None, None, :]                 # 1,1,S
    ray_off = np.array(ray_angles(cfg))[None, :, None]              # 1,R,1

    xs = np.fromiter((c.x for c in cars), float, n)[:, None, None]
    ys = np.fromiter((c.y for c in cars), float, n)[:, None, None]
    heads = np.fromiter((c.a for c in cars), float, n)[:, None, None]

    ang = heads + ray_off                                            # N,R,1
    dist = t * cfg.ray_range                                         # 1,1,S
    px = xs + np.cos(ang) * dist                                     # N,R,S
    py = ys + np.sin(ang) * dist

    on = track.on_track(px.ravel(), py.ravel()).reshape(n, cfg.n_rays, steps)
    off = ~on
    off[:, :, 0] = False  # the car's own position is always "on"
    first_off = np.argmax(off, axis=2)                               # N,R
    never_off = ~off.any(axis=2)
    hit_frac = np.where(never_off, 1.0, first_off / (steps - 1))
    proximity = 1.0 - hit_frac                                       # 0 clear .. ~1 touching

    speed = np.fromiter((c.v for c in cars), float, n) / cfg.max_speed
    speed = np.clip(speed, -0.3, 1.0)
    return np.column_stack([proximity, speed])


def step_all(cars, track, cfg):
    """Advance every alive car by one tick: sense, think, move, check."""
    alive = [c for c in cars if c.alive]
    if not alive:
        return
    senses = sense_all(alive, track, cfg)
    for c, row in zip(alive, senses):
        c.sensors = row.tolist()
        c.out = c.brain.step(c.sensors)

    n = len(alive)
    steer = np.fromiter((c.out[0] for c in alive), float, n)
    throttle = np.fromiter((c.out[1] for c in alive), float, n)
    v = np.fromiter((c.v for c in alive), float, n)
    a = np.fromiter((c.a for c in alive), float, n)
    x = np.fromiter((c.x for c in alive), float, n)
    y = np.fromiter((c.y for c in alive), float, n)

    accel = np.where(throttle >= 0, throttle * cfg.accel, throttle * cfg.brake)
    v = v + accel * cfg.dt
    v = v - v * cfg.drag * cfg.dt
    v = np.clip(v, -cfg.max_reverse, cfg.max_speed)

    turn_scale = np.minimum(1.0, np.abs(v) / cfg.min_turn_speed)
    a = a + steer * cfg.turn_rate * turn_scale * cfg.dt * np.sign(np.where(v == 0, 1, v))
    x = x + np.cos(a) * v * cfg.dt
    y = y + np.sin(a) * v * cfg.dt

    theta, _ = track.from_world(x, y)
    on = track.on_track(x, y)

    for i, c in enumerate(alive):
        c.v, c.a, c.x, c.y = float(v[i]), float(a[i]), float(x[i]), float(y[i])
        c.ticks += 1
        delta = float(theta[i]) - c.last_theta
        if delta > math.pi:
            delta -= 2 * math.pi
        elif delta < -math.pi:
            delta += 2 * math.pi
        c.theta_unwrapped += delta
        c.last_theta = float(theta[i])
        c.progress = float(track.progress(c.theta0 + c.theta_unwrapped)
                           - track.progress(c.theta0))
        c.best_progress = max(c.best_progress, c.progress)

        if c.progress >= track.lap_length:
            c.finished = True
            c.alive = False
            continue

        failed = False
        if not on[i]:
            failed = True
        elif abs(c.v) < cfg.stall_speed:
            c.stall += 1
            failed = c.stall > cfg.stall_ticks
        else:
            c.stall = 0

        if c.ticks >= cfg.max_ticks:
            c.alive = False
        elif failed:
            c.lives += 1
            c.respawn(track)


def fitness(car, track, cfg):
    """The best any single continuous attempt achieved this generation, not
    the sum across attempts - crashing on purpose to buy another attempt must
    never be a way to accumulate extra distance. Finishing a lap adds a big
    flat bonus, and a small speed bonus ranks two finishers by how quickly
    they did it."""
    base = max(0.0, car.best_progress)
    time_bonus = max(0.0, (cfg.max_ticks - car.ticks) * 0.02) if car.finished else 0.0
    return base + (cfg.lap_bonus if car.finished else 0.0) + time_bonus
