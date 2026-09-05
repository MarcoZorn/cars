"""Run with: python test_cars.py

Small and assert-based on purpose. Each check exists because the thing it
checks broke, or plausibly could have, at least once.
"""
import math
import random

import numpy as np

from car import Car, N_OUTPUTS, fitness, n_inputs, ray_angles, sense_all, step_all
from config import Config
from neat_core.brain import Brain
from neat_core.genes import Genome, Innovations
from neat_core.population import Population
from tracks import build_tracks


def genome(cfg, innov, n=40):
    g = Genome.minimal(n_inputs(cfg), N_OUTPUTS, innov, cfg)
    for _ in range(n):
        g.mutate(innov, cfg)
    return g


def test_ray_angles_cover_a_full_circle_without_duplicating_the_seam():
    """A 360-degree sensor ring has no edges, so the last ray must not repeat
    the first one - that would sense one direction twice and leave the true
    opposite of the front ray with no sensor at all."""
    cfg = Config()
    angles = ray_angles(cfg)
    assert len(angles) == cfg.n_rays
    assert angles[0] == -cfg.fov / 2
    spacing = 2 * math.pi / cfg.n_rays
    for i in range(1, len(angles)):
        assert abs((angles[i] - angles[i - 1]) - spacing) < 1e-9
    span = angles[-1] - angles[0]
    assert abs(span - (2 * math.pi - spacing)) < 1e-9, \
        "the last ray must stop one step short of wrapping onto the first"


def test_ray_angles_span_both_edges_of_an_open_cone():
    cfg = Config(n_rays=5, fov=2.0)
    angles = ray_angles(cfg)
    assert abs(angles[0] - (-1.0)) < 1e-9
    assert abs(angles[-1] - 1.0) < 1e-9


def test_tracks_are_simple_closed_curves():
    """A polar function is always simple by construction; this checks the
    guarantee actually holds after the anisotropic stretch and width offset,
    which is the part that could in principle break it."""
    cfg = Config()
    for t in build_tracks(cfg):
        on_center = t.on_track(t.center_pts[:, 0], t.center_pts[:, 1])
        assert on_center.all(), f"{t.name}: centerline must always be on the track"
        theta = t.theta_samples
        r = t.r_unit(theta)
        hw = t.half_width(theta)
        ox, oy = t.to_world(theta, r + hw * 1.4)
        assert not t.on_track(ox, oy).any(), \
            f"{t.name}: well outside the outer wall must read as off-track"
        ix, iy = t.to_world(theta, r * 0.3)
        assert not t.on_track(ix, iy).any(), \
            f"{t.name}: well inside the inner wall must read as off-track"


def test_tracks_do_not_pinch_shut():
    """The inner boundary must not come close to crossing itself at a tight
    bend, or a car could get walled in by geometry that looks fine from a
    distance."""
    cfg = Config()
    for t in build_tracks(cfg):
        pts = t.inner_pts
        n = len(pts)
        step = max(1, n // 180)
        min_gap = min(
            np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])[mask].min()
            for i in range(0, n, step)
            if (mask := (np.abs(np.arange(n) - i) > n * 0.05)
                & (np.abs(np.arange(n) - i - n) > n * 0.05)
                & (np.abs(np.arange(n) - i + n) > n * 0.05)).any()
        )
        narrowest = float(t.half_width(t.theta_samples).min())
        assert min_gap > narrowest * 0.5, \
            f"{t.name}: inner boundary comes close to crossing itself"


def test_difficulty_actually_makes_later_tracks_harder():
    """The curriculum only means anything if the geometry really changes -
    otherwise track 15 is just track 1 with a different name."""
    cfg = Config()
    tracks = build_tracks(cfg)
    easiest, hardest = tracks[0], tracks[-1]

    def wobble(t):
        theta = t.theta_samples
        r = t.r_unit(theta)
        return float(np.std(r) / t.base_r)

    def width_variation(t):
        return float(t.width_mult(t.theta_samples).std())

    assert wobble(hardest) > wobble(easiest), \
        "the hardest track's centerline should wobble more than the easiest"
    assert width_variation(hardest) >= width_variation(easiest), \
        "the hardest track should vary in width at least as much as the easiest"


def test_every_track_stays_wide_enough_for_the_car():
    """Even at its narrowest point, a track must leave real room either side
    of the car - not just barely wider than it."""
    cfg = Config()
    for t in build_tracks(cfg):
        narrowest = float(t.half_width(t.theta_samples).min())
        assert narrowest > cfg.car_w * 3, \
            f"{t.name}: a pinch point is too tight for the car to fit through"


def test_progress_accumulates_across_laps():
    cfg = Config()
    t = build_tracks(cfg)[0]
    p0 = t.progress(0.0)
    p1 = t.progress(2 * math.pi)      # exactly one full lap further
    p2 = t.progress(4 * math.pi)      # two full laps further
    assert abs((p1 - p0) - t.lap_length) < 1e-6
    assert abs((p2 - p1) - t.lap_length) < 1e-6


def test_sense_all_matches_per_car_geometry():
    """The batched ray march has to agree with directly checking the same
    points one at a time, or the speedup silently changes what a car sees."""
    cfg = Config()
    innov = Innovations()
    track = build_tracks(cfg)[0]
    random.seed(3)
    cars = [Car(genome(cfg, innov), None, track, cfg) for _ in range(5)]
    for c in cars:
        c.x += random.uniform(-20, 20)
        c.y += random.uniform(-20, 20)
        c.a += random.uniform(-1, 1)

    batched = sense_all(cars, track, cfg)
    angles = ray_angles(cfg)

    for c, row in zip(cars, batched):
        for i in range(cfg.n_rays):
            ang = c.a + angles[i]
            steps = 20
            hit = cfg.ray_range
            for k in range(1, steps):
                t = k / (steps - 1)
                d = t * cfg.ray_range
                px, py = c.x + math.cos(ang) * d, c.y + math.sin(ang) * d
                if not track.on_track(np.array([px]), np.array([py]))[0]:
                    hit = d
                    break
            expected = 1.0 - hit / cfg.ray_range
            assert abs(row[i] - expected) < 1e-9, \
                f"ray {i} disagrees: batched={row[i]:.4f} direct={expected:.4f}"


class _StillBrain:
    """A stub brain that always outputs zero, so this test exercises the
    stall-culling logic itself rather than whatever a random genome happens
    to do (which is not guaranteed to sit still, and depends on whatever
    state the global RNG was left in by earlier tests)."""
    inputs = bias = outputs = edges = act = depth = ()

    def step(self, values):
        return [0.0, 0.0]


def test_a_car_that_never_moves_scores_nothing():
    cfg = Config(max_ticks=50, stall_ticks=20)
    innov = Innovations()
    track = build_tracks(cfg)[0]
    g = genome(cfg, innov)
    c = Car(g, _StillBrain(), track, cfg)
    for _ in range(cfg.max_ticks):
        step_all([c], track, cfg)
        if not c.alive:
            break
    assert not c.alive, "a stationary car must be culled by the stall timer"
    assert not c.finished
    assert fitness(c, track, cfg) < 1.0


def test_crashing_respawns_instead_of_ending_the_race():
    """A crash ends the attempt, not the car's whole race: it should come back
    at the start line and keep trying until it finishes or time runs out."""
    cfg = Config(max_ticks=500)
    innov = Innovations()
    track = build_tracks(cfg)[0]
    g = genome(cfg, innov)
    c = Car(g, Brain(g), track, cfg)
    theta, r = track.from_world(*track.start_xy)
    ox, oy = track.to_world(theta, r + track.half_width(theta) * 3)
    c.x, c.y = float(ox), float(oy)
    lives_before = c.lives
    step_all([c], track, cfg)
    assert c.alive, "a crash must respawn the car, not end its race"
    assert c.lives == lives_before + 1
    assert (c.x, c.y) == track.start_xy, "a respawn must return to the start line"
    assert c.progress == 0.0, "the new attempt starts with no progress"
    assert not c.finished


def test_fitness_rewards_the_best_attempt_not_the_sum_of_attempts():
    """Crashing on purpose to buy another attempt must not be a way to
    accumulate extra distance - only the best single continuous attempt
    should count."""
    cfg = Config()
    innov = Innovations()
    track = build_tracks(cfg)[0]
    g = genome(cfg, innov)
    c = Car(g, Brain(g), track, cfg)
    c.best_progress = 300.0   # the high point of some earlier attempt
    c.progress = 40.0         # the current attempt has only gone this far
    assert fitness(c, track, cfg) < c.best_progress + 40.0, \
        "fitness must not simply add progress across separate attempts"
    assert fitness(c, track, cfg) >= c.best_progress


def test_evolution_produces_a_new_generation_with_history():
    cfg = Config(pop_size=40)
    pop = Population(n_inputs(cfg), N_OUTPUTS, cfg)
    for g in pop.genomes:
        g.fitness = random.random() * 100
    pop.evolve()
    assert pop.generation == 1
    assert len(pop.genomes) == cfg.pop_size
    assert len(pop.history) == 1
    assert pop.best is not None


def test_brain_output_is_bounded_and_deterministic():
    cfg = Config()
    innov = Innovations()
    g = genome(cfg, innov)
    b = Brain(g)
    x = [0.3, -0.2, 0.9, 0.0, 1.0, -0.7, 0.4, 0.6]
    a, b2 = b.step(x), b.step(x)
    assert a == b2
    assert all(-1.0 <= v <= 1.0 for v in a)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
