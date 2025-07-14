import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, Slider
import random
import math
from dataclasses import dataclass
import time

@dataclass
class Point:
    x: float
    y: float

    def distance(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)

class Ant:
    def __init__(self, position: Point, nest: Point):
        self.position = position
        self.nest = nest
        self.path = [position]
        self.energy = 100.0
        self.has_food = False

    def move(self, direction, distance):
        self.position.x += distance * math.cos(direction)
        self.position.y += distance * math.sin(direction)
        self.path.append(Point(self.position.x, self.position.y))
        self.energy -= 0.1

    def at_position(self, target: Point, threshold=1.5):
        return self.position.distance(target) < threshold

class PheromoneMap:
    def __init__(self, width, height, resolution=1):
        self.res = resolution
        self.grid = np.zeros((int(height / resolution), int(width / resolution)))
        self.decay = 0.95

    def deposit(self, pos: Point, strength):
        x, y = int(pos.x / self.res), int(pos.y / self.res)
        if 0 <= y < self.grid.shape[0] and 0 <= x < self.grid.shape[1]:
            self.grid[y, x] += strength

    def evaporate(self):
        self.grid *= self.decay

class AntSim:
    def __init__(self, width=50, height=50, n_ants=10):
        self.width, self.height = width, height
        self.nest = Point(width / 2, height / 2)
        self.food_sources = [Point(random.uniform(5, width-5), random.uniform(5, height-5)) for _ in range(5)]
        self.ants = [Ant(Point(self.nest.x, self.nest.y), self.nest) for _ in range(n_ants)]
        self.pheromones = PheromoneMap(width, height)
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.running = False
        self.speed = 1.0
        self._setup_ui()

    def _setup_ui(self):
        btn_ax = self.fig.add_axes([0.7, 0.02, 0.1, 0.04])
        self.btn = Button(btn_ax, 'Start/Stop')
        self.btn.on_clicked(self.toggle)

        slider_ax = self.fig.add_axes([0.2, 0.02, 0.4, 0.03])
        self.slider = Slider(slider_ax, 'Speed', 0.1, 2.0, valinit=1.0)
        self.slider.on_changed(lambda val: setattr(self, 'speed', val))

    def toggle(self, _):
        self.running = not self.running
        if self.running:
            self.run()

    def update_ants(self):
        for ant in self.ants:
            if ant.energy <= 0:
                continue

            if ant.has_food:
                direction = math.atan2(self.nest.y - ant.position.y, self.nest.x - ant.position.x)
                dist = min(1.0, ant.position.distance(self.nest))
                ant.move(direction, dist)
                self.pheromones.deposit(ant.position, 2.0)

                if ant.at_position(self.nest):
                    ant.has_food = False
                    ant.energy = 100.0
            else:
                best_score = -1
                best_angle = random.uniform(0, 2 * math.pi)
                for angle in np.linspace(0, 2 * math.pi, 12):
                    test_x = ant.position.x + 3 * math.cos(angle)
                    test_y = ant.position.y + 3 * math.sin(angle)
                    test_p = Point(test_x, test_y)
                    if not (0 < test_x < self.width and 0 < test_y < self.height):
                        continue
                    score = sum(10 / (0.1 + test_p.distance(f)) for f in self.food_sources if test_p.distance(f) < 5)
                    xg, yg = int(test_p.x), int(test_p.y)
                    if 0 <= yg < self.pheromones.grid.shape[0] and 0 <= xg < self.pheromones.grid.shape[1]:
                        score += self.pheromones.grid[yg, xg] * 0.5
                    if score > best_score:
                        best_score = score
                        best_angle = angle

                ant.move(best_angle, 1.0)
                if any(ant.at_position(f) for f in self.food_sources):
                    ant.has_food = True
                    self.pheromones.deposit(ant.position, 3.0)
                else:
                    self.pheromones.deposit(ant.position, 0.2)

        self.pheromones.evaporate()

    def draw(self):
        self.ax.clear()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)

        self.ax.imshow(self.pheromones.grid, origin='lower', extent=[0, self.width, 0, self.height], cmap='hot', alpha=0.5)

        for f in self.food_sources:
            self.ax.plot(f.x, f.y, 'go', markersize=8)

        self.ax.plot(self.nest.x, self.nest.y, 'bs', markersize=10)

        for ant in self.ants:
            xs, ys = zip(*[(p.x, p.y) for p in ant.path[-20:]])
            self.ax.plot(xs, ys, 'r-', linewidth=0.5)
            color = 'blue' if not ant.has_food else 'orange'
            self.ax.plot(ant.position.x, ant.position.y, 'o', color=color, markersize=4)

        self.ax.set_title("AntSLAM: Minimal Simulation")

    def run(self):
        while self.running:
            self.update_ants()
            self.draw()
            plt.pause(0.05 / self.speed)

sim = AntSim()
sim.draw()
plt.show()
