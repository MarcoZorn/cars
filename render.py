"""Drawing the track and the cars. Nothing here feeds back into the race."""
import colorsys
import math

import pygame

from tracks import CANVAS_H, CANVAS_W

BG = (12, 14, 19)
PANEL = (19, 22, 29)
LINE = (38, 43, 55)
TRACK = (54, 58, 70)
TEXT = (166, 176, 194)
DIM = (104, 114, 134)
BRIGHT = (233, 239, 249)
ACCENT = (122, 172, 255)
BEST_COLOR = (250, 200, 80)


def species_color(sid):
    h = 0.55 + 0.5 * ((sid * 0.381966) % 1.0)
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, 0.55, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


class Camera:
    def __init__(self, cfg):
        self.cfg = cfg
        self.x, self.y = CANVAS_W / 2, CANVAS_H / 2
        self.follow = None
        self.fit()

    def fit(self):
        self.zoom = min(self.cfg.view_w / CANVAS_W, self.cfg.view_h / CANVAS_H) * 0.96
        self.x, self.y = CANVAS_W / 2, CANVAS_H / 2
        self.follow = None

    def to_screen(self, wx, wy):
        return ((wx - self.x) * self.zoom + self.cfg.view_w / 2,
                (wy - self.y) * self.zoom + self.cfg.view_h / 2)

    def to_world(self, sx, sy):
        return ((sx - self.cfg.view_w / 2) / self.zoom + self.x,
                (sy - self.cfg.view_h / 2) / self.zoom + self.y)

    def zoom_at(self, sx, sy, factor):
        wx, wy = self.to_world(sx, sy)
        self.zoom = max(0.1, min(8.0, self.zoom * factor))
        nx, ny = self.to_world(sx, sy)
        self.x += wx - nx
        self.y += wy - ny

    def pan(self, dx, dy):
        self.follow = None
        self.x -= dx / self.zoom
        self.y -= dy / self.zoom

    def track_leader(self, car):
        if self.follow and car is not None:
            self.x += (car.x - self.x) * 0.12
            self.y += (car.y - self.y) * 0.12

    def scale(self, poly):
        return [self.to_screen(x, y) for x, y in poly]


class Renderer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.size = (cfg.view_w + cfg.panel_w, cfg.view_h + cfg.stats_h)
        self.screen = pygame.display.set_mode(self.size, pygame.RESIZABLE)
        pygame.display.set_caption("cars")
        self.cam = Camera(cfg)
        self.buttons = {}
        self.set_scale(cfg.ui_scale)

    def set_scale(self, s):
        self.cfg.ui_scale = max(0.6, min(2.6, s))
        n = self.cfg.ui_scale
        self.f = pygame.font.SysFont("monospace", int(14 * n))
        self.fb = pygame.font.SysFont("monospace", int(19 * n), bold=True)
        self.fs = pygame.font.SysFont("monospace", int(12 * n))
        self.lh = int(19 * n)

    def resize(self, w, h):
        want = (max(700, w), max(480, h))
        surf = pygame.display.get_surface()
        self.screen = surf if surf and surf.get_size() == want else \
            pygame.display.set_mode(want, pygame.RESIZABLE)
        self.size = self.screen.get_size()
        cfg = self.cfg
        cfg.panel_w = max(280, min(460, int(self.size[0] * 0.26)))
        cfg.stats_h = max(140, min(240, int(self.size[1] * 0.2)))
        cfg.view_w = self.size[0] - cfg.panel_w
        cfg.view_h = self.size[1] - cfg.stats_h
        self.cam = Camera(cfg)

    def button(self, label, x, y, w=None, on=False):
        n = self.cfg.ui_scale
        w = w or int(30 * n)
        h = int(26 * n)
        rect = pygame.Rect(int(x), int(y), w, h)
        pygame.draw.rect(self.screen, (34, 39, 50) if not on else (58, 88, 140), rect)
        pygame.draw.rect(self.screen, LINE, rect, 1)
        surf = self.f.render(label, True, BRIGHT if on else TEXT)
        self.screen.blit(surf, (rect.centerx - surf.get_width() // 2,
                                rect.centery - surf.get_height() // 2))
        self.buttons[label] = rect
        return rect.right + int(6 * n)

    def toolbar(self, y0, ui):
        n = self.cfg.ui_scale
        self.buttons = {}
        x = self.size[0] - int(430 * n)
        y = y0 + int(12 * n)
        self.text("text", x, y + int(4 * n), DIM, self.fs)
        x += int(42 * n)
        x = self.button("A-", x, y)
        x = self.button("A+", x, y)
        x += int(10 * n)
        self.text("zoom", x, y + int(4 * n), DIM, self.fs)
        x += int(50 * n)
        x = self.button("-", x, y)
        x = self.button("+", x, y)
        x = self.button("fit", x, y, int(48 * n))
        y2 = y + int(32 * n)
        x = self.size[0] - int(430 * n)
        x = self.button("pause" if not ui.get("paused") else "resume", x, y2, int(96 * n),
                        on=bool(ui.get("paused")))
        x = self.button(f"speed x{ui.get('speed', 1)}", x, y2, int(126 * n),
                        on=ui.get("speed", 1) > 1)
        self.button("follow", x, y2, int(84 * n), on=bool(ui.get("follow")))

    def text(self, s, x, y, col=TEXT, font=None):
        self.screen.blit((font or self.f).render(str(s), True, col), (x, y))

    def draw(self, track, cars, best, pop, ui):
        self.screen.fill(BG)
        self.cam.track_leader(best)
        self.dish(track, cars, best)
        self.brain_panel(best)
        self.stats(pop, ui)
        pygame.display.flip()

    def dish(self, track, cars, best):
        sc = self.screen
        sc.set_clip((0, 0, self.cfg.view_w, self.cfg.view_h))
        pygame.draw.polygon(sc, TRACK, self.cam.scale(track.outer_pts.tolist()))
        pygame.draw.polygon(sc, BG, self.cam.scale(track.inner_pts.tolist()))
        for i in range(0, len(track.center_pts), 40):
            x, y = self.cam.to_screen(*track.center_pts[i])
            pygame.draw.circle(sc, (30, 33, 42), (int(x), int(y)), 1)
        sx, sy = self.cam.to_screen(*track.start_xy)
        pygame.draw.circle(sc, (250, 220, 140), (int(sx), int(sy)), 5)

        for c in cars:
            if c.alive and c is not best:
                self.car(c, species_color(c.genome.species))
        if best is not None and best.alive:
            self.vision(best)
            self.car(best, BEST_COLOR, ring=True)
        sc.set_clip(None)
        pygame.draw.line(sc, LINE, (self.cfg.view_w, 0), (self.cfg.view_w, self.cfg.view_h))
        pygame.draw.line(sc, LINE, (0, self.cfg.view_h), (self.size[0], self.cfg.view_h))

    def _local_to_screen(self, c, lx, ly):
        """A point given in the car's own frame (world-scale units, x forward,
        y left), rotated by heading and placed at the car's world position."""
        ca, sa = math.cos(c.a), math.sin(c.a)
        wx = c.x + lx * ca - ly * sa
        wy = c.y + lx * sa + ly * ca
        return self.cam.to_screen(wx, wy)

    def car(self, c, color, ring=False):
        L, W = self.cfg.car_len, self.cfg.car_w
        # a small hull shape - tapered nose, square tail, a darker cockpit
        # stripe down the middle - rather than a single flat triangle
        body = [self._local_to_screen(c, *p) for p in (
            (L * 0.55, 0), (L * 0.15, W * 0.5), (-L * 0.45, W * 0.5),
            (-L * 0.45, -W * 0.5), (L * 0.15, -W * 0.5))]
        pygame.draw.polygon(self.screen, color, body)
        dark = tuple(max(0, v - 70) for v in color)
        cockpit = [self._local_to_screen(c, *p) for p in (
            (L * 0.1, W * 0.22), (-L * 0.25, W * 0.22),
            (-L * 0.25, -W * 0.22), (L * 0.1, -W * 0.22))]
        pygame.draw.polygon(self.screen, dark, cockpit)
        if ring:
            x, y = self.cam.to_screen(c.x, c.y)
            pygame.draw.circle(self.screen, BRIGHT, (int(x), int(y)),
                               max(3, int(L * 1.1 * self.cam.zoom)), 1)

    def vision(self, c):
        """The leader's sensor fan: one line per ray, coloured by how close
        the wall it is reporting actually is."""
        cfg = self.cfg
        half = cfg.fov / 2
        sx, sy = self.cam.to_screen(c.x, c.y)
        for i in range(cfg.n_rays):
            ang = c.a - half + cfg.fov * i / max(1, cfg.n_rays - 1)
            prox = c.sensors[i] if i < len(c.sensors) else 0.0
            length = cfg.ray_range * (1.0 - prox) * self.cam.zoom
            near = max(0.0, min(1.0, prox))
            col = (int(90 + 150 * near), int(200 - 120 * near), 90)
            ex = sx + math.cos(ang) * length
            ey = sy + math.sin(ang) * length
            pygame.draw.aaline(self.screen, col, (sx, sy), (ex, ey))
            pygame.draw.circle(self.screen, col, (int(ex), int(ey)), 2)

    def brain_panel(self, c):
        cfg = self.cfg
        x0 = cfg.view_w
        pygame.draw.rect(self.screen, PANEL, (x0, 0, cfg.panel_w, cfg.view_h))
        pad = int(16 * cfg.ui_scale)
        self.text("LEADER", x0 + pad, pad, BRIGHT, self.fb)
        if c is None:
            self.text("no car yet", x0 + pad, pad + self.lh * 2, DIM, self.fs)
            return
        n, e = c.genome.complexity()
        rows = [
            f"species {c.genome.species}   {n}n / {e}c",
            f"progress {c.progress:.0f}   best {c.best_progress:.0f}",
            f"attempt #{c.lives}   {'FINISHED' if c.finished else 'racing'}",
            f"speed {c.v:.0f}   ticks {c.ticks}",
            f"steer {c.out[0]:+.2f}   throttle {c.out[1]:+.2f}",
        ]
        for i, r in enumerate(rows):
            self.text(r, x0 + pad, pad + self.lh * (1.4 + i * 0.95), TEXT, self.fs)

        top = pad + int(self.lh * 6.8)
        bottom = cfg.view_h - int(30 * cfg.ui_scale)
        b = c.brain
        cols = {}
        for node, d in b.depth.items():
            cols.setdefault(d, []).append(node)
        maxd = max(cols) or 1
        pos = {}
        lw = int(70 * cfg.ui_scale)
        for d, nodes in cols.items():
            nodes.sort()
            for i, node in enumerate(nodes):
                x = x0 + lw + (cfg.panel_w - lw - int(60 * cfg.ui_scale)) * (d / maxd)
                y = top + (bottom - top) * ((i + 0.5) / len(nodes))
                pos[node] = (x, y)
        for src, dst, w in b.edges:
            if src in pos and dst in pos:
                sig = b.act.get(src, 0.0) * w
                col = self._act_color(sig)
                pygame.draw.aaline(self.screen, col, pos[src], pos[dst])
        for node, (x, y) in pos.items():
            v = b.act.get(node, 0.0)
            r = (4 + 4 * min(1.0, abs(v))) * cfg.ui_scale
            pygame.draw.circle(self.screen, self._act_color(v), (int(x), int(y)), int(r))
        names = [f"ray{i}" for i in range(cfg.n_rays)] + ["speed"]
        for node, name in zip(b.inputs, names):
            x, y = pos[node]
            self.text(name, x - int(56 * cfg.ui_scale), y - 7, DIM, self.fs)
        for node in b.bias:
            if node in pos:
                x, y = pos[node]
                self.text("bias", x - int(56 * cfg.ui_scale), y - 7, DIM, self.fs)
        for node, name in zip(b.outputs, ["steer", "throttle"]):
            if node in pos:
                x, y = pos[node]
                self.text(name, x + int(10 * cfg.ui_scale), y - 7, BRIGHT, self.fs)

    def _act_color(self, v):
        v = max(-1.0, min(1.0, v))
        if v >= 0:
            return int(70 + 185 * v), int(70 + 100 * v), int(80 - 40 * v)
        return int(70 + 20 * v), int(70 + 40 * -v), int(80 + 175 * -v)

    def stats(self, pop, ui):
        cfg = self.cfg
        y0 = cfg.view_h
        pygame.draw.rect(self.screen, PANEL, (0, y0, self.size[0], cfg.stats_h))
        pad = int(14 * cfg.ui_scale)
        self.text(f"gen {pop.generation}/{cfg.generations}   {ui.get('track_name','')}",
                  pad, y0 + pad, BRIGHT, self.fb)
        rows = [
            f"alive    {ui.get('alive', 0)}/{cfg.pop_size}",
            f"tick     {ui.get('tick', 0)}/{cfg.max_ticks}",
            f"species  {len(pop.species)}",
            f"best ever      {pop.best.fitness:.0f}" if pop.best else "best ever      -",
            f"best this gen  {ui.get('gen_best', 0):.0f}",
        ]
        for i, r in enumerate(rows):
            self.text(r, pad, y0 + pad + self.lh * (1.3 + i * 0.85), TEXT, self.fs)
        # the toolbar owns the right edge; the graph gets whatever is left
        text_w = int(230 * cfg.ui_scale)
        toolbar_w = int(450 * cfg.ui_scale)
        gx = pad + text_w
        gw = max(120, self.size[0] - gx - toolbar_w - pad)
        self.graph(pop.history, gx, y0 + pad, gw, cfg.stats_h - pad * 2)
        self.toolbar(y0, ui)
        hint = ("space pause  f speed  +/- zoom  0 fit  drag pan  "
                "c follow  [ ] text  q quit")
        self.text(hint[: max(20, int((self.size[0] - int(470 * cfg.ui_scale))
                                     / (7.2 * cfg.ui_scale)))],
                  pad, y0 + cfg.stats_h - int(18 * cfg.ui_scale), (92, 100, 118), self.fs)

    def graph(self, hist, x, y, w, h):
        pygame.draw.rect(self.screen, BG, (x, y, w, h))
        self.text("fitness / generation", x + 6, y + 3, DIM, self.fs)
        if len(hist) < 2:
            return
        top = max(max(r["best"] for r in hist), 1.0)
        n = len(hist)
        for key, col in (("mean", (110, 120, 150)), ("best", ACCENT)):
            pts = [(x + w * i / (n - 1), y + h - 5 - (h - 22) * (r[key] / top))
                   for i, r in enumerate(hist)]
            pygame.draw.aalines(self.screen, col, False, pts)
