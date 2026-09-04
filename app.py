"""Interactive loop: race a generation, evolve, move to the next track, repeat.

Tracks are cycled in order across generations - by 250 generations the
population has been raced on each of the 15 roughly seventeen times, which
pushes toward a driver that generalises rather than one that has memorised a
single layout.
"""
import pygame

from car import Car, N_OUTPUTS, fitness, n_inputs, step_all
from config import Config
from neat_core.brain import Brain
from neat_core.population import Population
from render import Renderer
from tracks import build_tracks


class App:
    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        pygame.init()
        self.fit_to_screen()
        self.render = Renderer(self.cfg)
        self.tracks = build_tracks(self.cfg)
        self.pop = Population(n_inputs(self.cfg), N_OUTPUTS, self.cfg)
        self.clock = pygame.time.Clock()
        self.paused = False
        self.fast = 1
        self.drag = None
        self.track = self.tracks[0]
        self.new_generation()

    def fit_to_screen(self):
        info = pygame.display.Info()
        w = int(info.current_w * 0.82)
        h = int(info.current_h * 0.78)
        if not self.cfg.ui_scale:
            self.cfg.ui_scale = max(1.0, min(3.0, info.current_h / 900.0))
        self.cfg.panel_w = max(280, min(460, int(w * 0.26)))
        self.cfg.stats_h = max(140, min(240, int(h * 0.2)))
        self.cfg.view_w = w - self.cfg.panel_w
        self.cfg.view_h = h - self.cfg.stats_h

    def new_generation(self):
        self.pop.speciate()
        self.track = self.tracks[self.pop.generation % len(self.tracks)]
        self.cars = [Car(g, Brain(g), self.track, self.cfg) for g in self.pop.genomes]
        self.tick = 0

    def leader(self):
        alive = [c for c in self.cars if c.alive]
        pool = alive or self.cars
        return max(pool, key=lambda c: fitness(c, self.track, self.cfg))

    def toolbar_click(self, pos):
        hit = next((k for k, r in self.render.buttons.items() if r.collidepoint(pos)), None)
        if hit is None:
            return False
        cam = self.render.cam
        mid = (self.cfg.view_w / 2, self.cfg.view_h / 2)
        if hit == "A-":
            self.render.set_scale(self.cfg.ui_scale - 0.15)
        elif hit == "A+":
            self.render.set_scale(self.cfg.ui_scale + 0.15)
        elif hit == "-":
            cam.zoom_at(*mid, 1 / 1.35)
        elif hit == "+":
            cam.zoom_at(*mid, 1.35)
        elif hit == "fit":
            cam.fit()
        elif hit in ("pause", "resume"):
            self.paused = not self.paused
        elif hit.startswith("speed"):
            self.fast = {1: 4, 4: 16, 16: 1}[self.fast]
        elif hit == "follow":
            cam.follow = None if cam.follow else self.leader()
        return True

    def events(self):
        cam = self.render.cam
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            if e.type == pygame.VIDEORESIZE:
                self.render.resize(e.w, e.h)
            elif e.type == pygame.WINDOWRESIZED:
                self.render.resize(e.x, e.y)
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and self.toolbar_click(e.pos):
                continue
            if e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 4:
                    cam.zoom_at(*e.pos, 1.15)
                elif e.button == 5:
                    cam.zoom_at(*e.pos, 1 / 1.15)
                elif e.pos[0] < self.cfg.view_w:
                    self.drag = e.pos
            if e.type == pygame.MOUSEBUTTONUP:
                self.drag = None
            if e.type == pygame.MOUSEMOTION and self.drag:
                cam.pan(e.rel[0], e.rel[1])
            if e.type == pygame.MOUSEWHEEL:
                cam.zoom_at(*pygame.mouse.get_pos(), 1.15 ** e.y)
            if e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_q, pygame.K_ESCAPE):
                    return False
                if e.key == pygame.K_SPACE:
                    self.paused = not self.paused
                if e.key == pygame.K_f:
                    self.fast = {1: 4, 4: 16, 16: 1}[self.fast]
                if e.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    cam.zoom_at(self.cfg.view_w / 2, self.cfg.view_h / 2, 1.35)
                if e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    cam.zoom_at(self.cfg.view_w / 2, self.cfg.view_h / 2, 1 / 1.35)
                if e.key == pygame.K_0:
                    cam.fit()
                if e.key == pygame.K_c:
                    cam.follow = None if cam.follow else self.leader()
                if e.key == pygame.K_LEFTBRACKET:
                    self.render.set_scale(self.cfg.ui_scale - 0.15)
                if e.key == pygame.K_RIGHTBRACKET:
                    self.render.set_scale(self.cfg.ui_scale + 0.15)
        return True

    def race_finished(self):
        return self.tick >= self.cfg.max_ticks or not any(c.alive for c in self.cars)

    def run(self):
        running = True
        while running:
            running = self.events()
            if not self.paused:
                for _ in range(self.fast):
                    if self.race_finished():
                        break
                    step_all(self.cars, self.track, self.cfg)
                    self.tick += 1
            if self.race_finished():
                for c in self.cars:
                    c.genome.fitness = fitness(c, self.track, self.cfg)
                self.pop.evolve()
                self.new_generation()
                if self.pop.generation >= self.cfg.generations:
                    self.paused = True
            leader = self.leader()
            if self.render.cam.follow is not None:
                self.render.cam.follow = leader
            gen_best = max((fitness(c, self.track, self.cfg) for c in self.cars), default=0.0)
            self.render.draw(self.track, self.cars, leader, self.pop, {
                "alive": sum(c.alive for c in self.cars),
                "tick": self.tick,
                "track_name": self.track.name,
                "gen_best": gen_best,
                "fast": self.fast if self.fast > 1 else 0,
                "speed": self.fast,
                "paused": self.paused,
                "follow": self.render.cam.follow is not None,
            })
            self.clock.tick(0 if self.fast > 1 else 30)
        pygame.quit()
