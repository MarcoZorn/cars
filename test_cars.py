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
from neat_core.genes import HIDDEN, Genome, Innovations
from neat_core.population import Population, Species
from tracks import Track, build_tracks


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


def test_track_room_keeps_pinches_passable_at_hardness_two():
    """With track_room the hard set keeps three car widths of half-width
    everywhere; without it the same set does pinch, so the test can fail."""
    theta = np.linspace(-np.pi, np.pi, 2000)

    def narrowest(room):
        cfg = Config()
        cfg.track_hardness, cfg.track_room = 2.0, room
        return min(float(t.half_width(theta).min()) for t in build_tracks(cfg)), cfg.car_w * 3

    got, need = narrowest(True)
    assert got > need, f"track_room: narrowest half-width {got:.2f} is under {need:.2f}"
    assert narrowest(False)[0] < need, "without track_room the hard set should pinch"


def test_difficulty_actually_makes_later_tracks_harder():
    """The curriculum only means anything if the geometry really changes -
    otherwise the last track is just the first with a different name.

    Compared a third of the set at a time, not first against last: any two
    tracks are different random shapes. The centerline itself hardly ramps
    (more harmonics, each smaller, so about the same overall wobble); what
    does is the number of bends and the width chicanes."""
    cfg = Config()
    tracks = build_tracks(cfg)
    k = len(tracks) // 3
    easy, hard = tracks[:k], tracks[-k:]

    def mean(f, ts):
        return float(np.mean([f(t) for t in ts]))

    def bends(t):
        return len(t.harmonics)

    def width_variation(t):
        return float(t.width_mult(t.theta_samples).std())

    assert mean(bends, hard) > mean(bends, easy), \
        "the hardest tracks should have more bends than the easiest"
    assert mean(width_variation, hard) > mean(width_variation, easy), \
        "the hardest tracks should vary in width more than the easiest"


def test_a_mirrored_track_is_the_same_track_reflected():
    """mirror=True reflects the track top to bottom, so it is driven the
    other way round; every point on its centerline is the original's,
    reflected, and the car sets off in the reflected direction."""
    a, b = Track(321, difficulty=0.6), Track(321, difficulty=0.6, mirror=True)
    pa, pb = np.asarray(a.center_pts), np.asarray(b.center_pts)
    assert np.allclose(pa[:, 0], pb[:, 0]), "the same x everywhere"
    assert np.allclose(pa[:, 1] - a.cy, -(pb[:, 1] - b.cy)), "y reflected about the centre"
    assert abs(math.sin(a.start_heading) + math.sin(b.start_heading)) < 1e-6, "setting off reflected"


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


def test_stagnation_never_culls_the_only_improving_species():
    """With one species still improving and the rest stale, the cull keeps a
    stale runner-up for diversity - it must never keep two stale species and
    throw the improving one away."""
    cfg = Config(pop_size=30, stagnation=3)
    pop = Population(n_inputs(cfg), N_OUTPUTS, cfg)
    gs = pop.genomes
    species = []
    for i in range(3):
        s = Species(gs[i * 10])
        s.members = gs[i * 10:(i + 1) * 10]
        for g in s.members:
            g.fitness = float(i + 1)
        s.best, s.stale = 100.0, 10
        species.append(s)
    fresh = species[0]
    fresh.best, fresh.stale = 0.0, 0
    pop.species = list(species)
    pop._cull()
    assert fresh in pop.species, "the only improving species was culled"
    assert species[2] in pop.species, "the strongest stale species should be kept"
    assert len(pop.species) == 2


def test_brain_output_is_bounded_and_deterministic():
    cfg = Config()
    innov = Innovations()
    g = genome(cfg, innov)
    b = Brain(g)
    x = [0.3, -0.2, 0.9, 0.0, 1.0, -0.7, 0.4, 0.6, 0.5][:n_inputs(cfg)]
    a, b2 = b.step(x), b.step(x)
    assert a == b2
    assert all(-1.0 <= v <= 1.0 for v in a)



def test_brain_evaluates_hidden_nodes_before_what_they_feed():
    """A hand-built in -> hidden -> out chain must produce exactly the nested
    tanh - if the evaluation order is wrong the output reads a stale 0."""
    cfg = Config()
    innov = Innovations()
    g = Genome.minimal(n_inputs(cfg), N_OUTPUTS, innov, cfg)
    for c in g.conns.values():
        c.w = 0.0
    src = sorted(g.ids("in"))[0]
    out = sorted(g.ids("out"))[0]
    h = innov.node()
    g.nodes[h] = HIDDEN
    g.add_conn(src, h, 2.0, innov)
    g.add_conn(h, out, -1.5, innov)
    x = [0.4] + [0.0] * (n_inputs(cfg) - 1)
    got = Brain(g).step(x)[0]
    assert abs(got - math.tanh(-1.5 * math.tanh(2.0 * 0.4))) < 1e-12


def test_mutation_never_makes_a_cycle():
    """The brain is feedforward only: after heavy structural mutation every
    node must still land in the topological order."""
    cfg = Config(p_add_conn=0.8, p_add_node=0.4)
    innov = Innovations()
    random.seed(5)
    for _ in range(20):
        g = genome(cfg, innov, n=60)
        assert len(Brain(g).order) == len(g.nodes)


def test_wall_lookup_matches_the_analytic_walls():
    """on_track reads the walls from a table; just inside a wall must still
    be on the track and just outside must be off, everywhere around."""
    cfg = Config()
    for t in build_tracks(cfg):
        theta = np.linspace(-math.pi, math.pi, 5000)
        r, hw = t.r_unit(theta), t.half_width(theta)
        for edge in (r + hw, r - hw):
            sign = np.sign(edge - r)
            assert t.on_track(*t.to_world(theta, edge - sign * 0.05)).all(), t.name
            assert not t.on_track(*t.to_world(theta, edge + sign * 0.05)).any(), t.name



def test_pillars_block_one_side_and_leave_the_other_open():
    """A pillar is solid, the start line stays clear, and at every pillar
    there is still a lane past it wide enough for the car (the standard set)."""
    cfg = Config(track_obstacles=6, track_hardness=1.0)
    tracks = build_tracks(cfg)
    assert not tracks[0].pillars and len(tracks[-1].pillars) == 6
    for t in tracks:
        assert t.on_track(*t.start_xy), f"{t.name}: start line blocked"
        for px, py, rad in t.pillars:
            assert not t.on_track(px, py)
            theta, r = t.from_world(px, py)
            rc, hw = t.r_unit(theta), t.half_width(theta)
            side = 1 if r > rc else -1
            # the open side: from the far wall to the pillar's edge
            lane = np.linspace(rc - side * hw * 0.9, rc - side * hw * 0.05, 30)
            assert t.on_track(*t.to_world(np.full(30, theta), lane)).all(), \
                f"{t.name}: no lane left past a pillar"
            assert hw * 0.85 * min(t.sx, t.sy) > cfg.car_w * 2



def test_the_default_hard_set_leaves_a_lane_past_every_pillar():
    """The default tracks are harder and narrower, but past every pillar the
    open side is still one and a half car widths at least."""
    cfg = Config()
    for t in build_tracks(cfg):
        for px, py, rad in t.pillars:
            theta, _ = t.from_world(px, py)
            assert t.half_width(theta) * 0.85 * min(t.sx, t.sy) > cfg.car_w * 1.5, f"{t.name}: lane too narrow"

def test_a_save_carries_on_exactly():
    import os
    import tempfile
    from neat_core.population import load, save
    random.seed(11)
    cfg = Config(pop_size=40)
    pop = Population(n_inputs(cfg), N_OUTPUTS, cfg)
    for g in pop.genomes:
        g.fitness = random.random() * 100
    pop.evolve()
    path = os.path.join(tempfile.mkdtemp(), "pop.pkl.gz")
    save(pop, path)

    def next_gen(p):
        for g in p.genomes:
            g.fitness = random.random() * 100
        p.evolve()
        return [sorted((k, c.w, c.enabled) for k, c in g.conns.items()) for g in p.genomes]

    a = next_gen(pop)
    b = next_gen(load(path))
    assert a == b, "a reloaded population must evolve exactly as the original"



def test_a_delayed_car_acts_on_old_outputs():
    cfg = Config(delay=3)
    track = build_tracks(cfg)[0]
    pop = Population(n_inputs(cfg), N_OUTPUTS, cfg)
    g = pop.genomes[0]
    c = Car(g, Brain(g), track, cfg)
    said = []
    real = c.brain.step
    c.brain.step = lambda s: said.append(list(real(s))) or said[-1]
    acted = []
    for _ in range(6):
        step_all([c], track, cfg)
        acted.append(c.out)
    assert acted[:3] == [[0.0, 0.0]] * 3 and acted[3:] == said[:3]



def test_bootstrap_interval_brackets_the_mean():
    from bench import bootstrap
    lo, hi = bootstrap([9, 10, 11, 12, 13, 14])
    assert lo < 11.5 < hi and 9 <= lo and hi <= 14
    assert bootstrap([7, 7, 7]) == (7.0, 7.0)


def test_report_reads_a_bench_log():
    import os
    import tempfile
    from report import page, parse
    d = tempfile.mkdtemp()
    p = os.path.join(d, "x.log")
    with open(p, "w") as f:
        f.write("n_rays=8  pop=200 gens=20 seeds=3  (99s)\n"
                "  train    laps/30 per seed: 30 29 30\n"
                "  train    laps mean 29.67  lap-frac mean 0.99  lap ticks mean 220\n"
                "  held-out laps/15 per seed: 14 12 13\n"
                "  held-out laps mean 13.00  lap-frac mean 0.96  lap ticks mean 215\n")
    r = parse(p)
    assert r["set"] == "n_rays=8" and r["seeds"] == 3 and r["held-out"]["laps"] == [14, 12, 13]
    assert r["mean"] == 13.0 and r["lo"] <= 13.0 <= r["hi"] and r["held-out"]["ticks"] == 215
    assert "n_rays=8" in page([r])
    with open(os.path.join(d, "junk.log"), "w") as f:
        f.write("not a bench run\n")
    assert parse(os.path.join(d, "junk.log")) is None
    with open(p, "w") as f:
        f.write("n_rays=8  exam: track_obstacles=3  pop=200 gens=20 seeds=1  (9s)\n  held-out laps/15 per seed: 9\n")
    assert parse(p)["set"] == "n_rays=8  exam: track_obstacles=3"

def test_report_charts_a_swept_setting():
    import os
    import tempfile
    from report import chart, page, parse, sweeps
    d = tempfile.mkdtemp()
    runs = []
    for i, (rays, laps) in enumerate([(4, "8 9 7"), (8, "11 12 10"), (16, "13 14 12")]):
        p = os.path.join(d, f"r{rays}.log")
        with open(p, "w") as f:
            f.write(f"n_rays={rays}  pop=200 gens=20 seeds=3  (9s)\n  held-out laps/15 per seed: {laps}\n")
        os.utime(p, (1000 + i, 1000 + i))
        runs.append(parse(p))
    lone = os.path.join(d, "x.log")
    with open(lone, "w") as f:
        f.write("delay=2  pop=200 gens=20 seeds=3  (9s)\n  held-out laps/15 per seed: 5 5 5\n")
    runs.append(parse(lone))
    sw = sweeps(runs)
    assert list(sw) == ["n_rays"] and [r["set"] for r in sw["n_rays"]] == ["n_rays=4", "n_rays=8", "n_rays=16"], "one sweep, in order of value; delay=2 alone is not one"
    svg = chart("n_rays", sw["n_rays"])
    assert svg.count("<circle") == 3 and svg.count("stroke-width=\"3\"") == 3, "a point and an interval per value"
    assert "<h2>n_rays</h2>" in page(runs) and "<h2>delay</h2>" not in page(runs)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} passed")


def test_evolving_eyes_are_copied_mutated_and_seen():
    """With evolve_eyes each genome owns its ray turns and range: a copy or a
    crossover child must not share them with its parent, mutation must move
    them within bounds, and the ray march must look where they say."""
    from neat_core.population import Population
    cfg = Config(evolve_eyes=True, pop_size=30)
    random.seed(5)
    pop = Population(n_inputs(cfg), N_OUTPUTS, cfg)
    g = pop.genomes[0]
    assert g.eyes is not None and len(g.eyes[0]) == cfg.n_rays
    c = g.copy()
    c.eyes[0][0] += 1.0
    assert g.eyes[0][0] != c.eyes[0][0], "a copy must not share its parent's eyes"
    for _ in range(200):
        g.mutate(pop.innov, cfg)
    assert any(t != 0 for t in g.eyes[0]) and cfg.eye_min <= g.eyes[1] <= cfg.eye_max
    # zero turns and the stock range see exactly what the fixed fan sees
    track = build_tracks(cfg)[0]
    plain = Config()
    a = Car(genome(plain, Innovations()), None, track, plain)
    b = Car(pop.genomes[1], None, track, cfg)
    b.genome.eyes = ([0.0] * cfg.n_rays, cfg.ray_range)
    assert np.allclose(sense_all([a], track, plain), sense_all([b], track, cfg))
    # and a car that looks only straight back sees the same on every ray
    b.genome.eyes = ([math.pi - x for x in ray_angles(cfg)], cfg.ray_range)
    row = sense_all([b], track, cfg)[0][:cfg.n_rays]
    assert np.allclose(row, row[0])
