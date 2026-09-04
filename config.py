from dataclasses import dataclass


@dataclass
class Config:
    # population
    pop_size: int = 1000
    generations: int = 250
    elitism: int = 2
    survival: float = 0.4
    p_crossover: float = 0.75
    stagnation: int = 20
    target_species: int = 16

    # speciation
    compat_threshold: float = 3.0
    c_disjoint: float = 1.0
    c_weight: float = 0.5
    small_genome: int = 20

    # mutation
    p_weight: float = 0.8
    p_weight_replace: float = 0.1
    weight_step: float = 0.35
    weight_init_std: float = 1.0
    weight_cap: float = 8.0
    p_add_conn: float = 0.14
    p_add_node: float = 0.05
    p_toggle: float = 0.01
    p_inherit_disabled: float = 0.75
    add_conn_tries: int = 20

    # sensors: rays fanned across the windshield, plus own speed
    n_rays: int = 7
    fov: float = 2.5              # radians, total field of view
    ray_range: float = 220.0

    # car physics (arcade model, not a real bicycle model)
    accel: float = 220.0          # px/s^2 at full throttle
    brake: float = 320.0          # px/s^2 at full reverse throttle
    drag: float = 0.6             # fraction of speed lost per second
    max_speed: float = 260.0
    max_reverse: float = 60.0
    turn_rate: float = 3.2        # rad/s at max steering and max speed
    min_turn_speed: float = 18.0  # below this, steering barely does anything
    car_len: float = 14.0
    car_w: float = 7.0

    # race
    max_ticks: int = 1400         # timeout if a car never crashes
    dt: float = 1 / 30
    stall_ticks: int = 90         # crash if slower than this speed this long
    stall_speed: float = 8.0

    # fitness: reward covering ground fast, punish idling
    lap_bonus: float = 4000.0

    # tracks
    n_tracks: int = 15
    track_seed: int = 100         # tracks are fixed, independent of run seed

    # view
    view_w: int = 1200
    view_h: int = 860
    panel_w: int = 380
    stats_h: int = 190
    ui_scale: float = 0.0         # 0 = pick from desktop resolution
