"""Procedurally generated closed-loop tracks.

Each track's centerline is a polar function r(theta) around a fixed centre,
optionally stretched by an anisotropic (x, y) scale. A polar function is
always a simple closed curve - exactly one point per angle - so no matter how
wild the harmonics are, the loop can never cross itself. Stretching it with a
positive (sx, sy) scale is an affine map and preserves that property, which is
what turns a plain wobbly circle into ovals, figure-eight-ish sweeps and tight
chicanes without ever risking a self-intersecting track.

Because every boundary and every physics check is derived from the same two
analytic functions - r(theta) and half_width(theta) - a car's position can be
tested against the track in O(1) by inverting the affine map and looking the
angle up, rather than testing against a polygon with hundreds of edges. That
is what makes checking a few thousand sensor rays per tick affordable.
"""
import math
import random

import numpy as np

CANVAS_W = 1400
CANVAS_H = 1000
SAMPLES = 1440  # centerline resolution, used for rendering and arc length


class Track:
    def __init__(self, seed, name=None):
        rng = random.Random(seed)
        self.seed = seed
        self.name = name or f"Track {seed}"
        self.cx, self.cy = CANVAS_W / 2, CANVAS_H / 2

        self.base_r = rng.uniform(210, 320)
        n_harm = rng.randint(2, 4)
        self.harmonics = [
            (rng.randint(2, 6), rng.uniform(0.05, 0.22) / n_harm, rng.uniform(0, 2 * math.pi))
            for _ in range(n_harm)
        ]
        # anisotropic stretch: ovals and elongated loops, never a self-cross
        self.sx = rng.uniform(0.75, 1.35)
        self.sy = rng.uniform(0.75, 1.35)
        self.width_frac = rng.uniform(0.10, 0.15)

        self._build()

    # --- the analytic track shape ---

    def r_unit(self, theta):
        theta = np.asarray(theta, dtype=float)
        r = np.full_like(theta, self.base_r)
        for k, amp, phase in self.harmonics:
            r = r + self.base_r * amp * np.sin(k * theta + phase)
        return np.maximum(r, self.base_r * 0.35)

    def half_width(self, theta):
        return self.r_unit(theta) * self.width_frac

    def to_world(self, theta, r):
        theta = np.asarray(theta, dtype=float)
        r = np.asarray(r, dtype=float)
        x = self.cx + r * np.cos(theta) * self.sx
        y = self.cy + r * np.sin(theta) * self.sy
        return x, y

    def from_world(self, x, y):
        x0 = (np.asarray(x, dtype=float) - self.cx) / self.sx
        y0 = (np.asarray(y, dtype=float) - self.cy) / self.sy
        theta = np.arctan2(y0, x0)
        r = np.hypot(x0, y0)
        return theta, r

    def on_track(self, x, y):
        theta, r = self.from_world(x, y)
        rc = self.r_unit(theta)
        hw = rc * self.width_frac
        return (r >= rc - hw) & (r <= rc + hw)

    # --- precomputed geometry, arc length, start line ---

    def _build(self):
        theta = np.linspace(-math.pi, math.pi, SAMPLES, endpoint=False)
        r = self.r_unit(theta)
        hw = r * self.width_frac
        self.theta_samples = theta
        cx, cy = self.to_world(theta, r)
        self.center_pts = np.column_stack([cx, cy])
        ox, oy = self.to_world(theta, r + hw)
        ix, iy = self.to_world(theta, r - hw)
        self.outer_pts = np.column_stack([ox, oy])
        self.inner_pts = np.column_stack([ix, iy])

        seg = np.hypot(np.diff(cx, append=cx[0]), np.diff(cy, append=cy[0]))
        self.arc_len = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
        self.lap_length = float(np.sum(seg))

        # start/finish at theta = -pi, facing the direction of increasing theta
        i0 = 0
        i1 = 1
        self.start_xy = (float(cx[i0]), float(cy[i0]))
        self.start_heading = math.atan2(cy[i1] - cy[i0], cx[i1] - cx[i0])

    def progress(self, theta_unwrapped):
        """Arc-length-ish distance travelled, given an unwrapped theta (can
        exceed +-pi to represent multiple laps)."""
        laps = np.floor((theta_unwrapped + math.pi) / (2 * math.pi))
        wrapped = theta_unwrapped - laps * 2 * math.pi
        idx = ((wrapped + math.pi) / (2 * math.pi) * SAMPLES).astype(np.int64) % SAMPLES
        return laps * self.lap_length + self.arc_len[idx]


def build_tracks(cfg):
    return [Track(cfg.track_seed + i, name=f"Track {i + 1}") for i in range(cfg.n_tracks)]
