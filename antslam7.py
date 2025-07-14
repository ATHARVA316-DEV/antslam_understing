import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import random
import math
from dataclasses import dataclass

# === Data Classes ===
@dataclass
class Point:
    x: float
    y: float

    def distance_to(self, other: 'Point') -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

# === Ant Class ===
class Ant:
    def __init__(self, position: Point):
        self.position = position
        self.path = [position]
        self.speed = 1.0
        self.carrying_food = False

    def move(self, direction: float):
        dx = self.speed * math.cos(direction)
        dy = self.speed * math.sin(direction)
        new_position = Point(self.position.x + dx, self.position.y + dy)
        self.position = new_position
        self.path.append(new_position)

    def deposit_pheromone(self, pheromone_map):
        pheromone_map.add_pheromone(self.position, strength=1.0)

# === Pheromone Map ===
class PheromoneMap:
    def __init__(self, width, height, resolution=1.0):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.grid_w = int(width / resolution)
        self.grid_h = int(height / resolution)
        self.grid = np.zeros((self.grid_h, self.grid_w))
        self.decay = 0.98

    def add_pheromone(self, pos: Point, strength: float):
        x = int(pos.x / self.resolution)
        y = int(pos.y / self.resolution)
        if 0 <= x < self.grid_w and 0 <= y < self.grid_h:
            self.grid[y, x] += strength

    def evaporate(self):
        self.grid *= self.decay

# === Environment and Simulation ===
class AntSLAMSingle:
    def __init__(self, width=50, height=50):
        self.width = width
        self.height = height
        self.pheromone_map = PheromoneMap(width, height)
        self.ant = Ant(Point(width // 2, height // 2))
        self.food_sources = self.generate_food_sources(5)
        self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 6))
        self.running = True

    def generate_food_sources(self, count):
        return [Point(random.randint(5, self.width - 5), random.randint(5, self.height - 5)) for _ in range(count)]

    def simulate_step(self):
        best_angle = random.uniform(0, 2 * math.pi)
        min_dist = float('inf')
        for food in self.food_sources:
            d = self.ant.position.distance_to(food)
            if d < min_dist:
                min_dist = d
                best_angle = math.atan2(food.y - self.ant.position.y, food.x - self.ant.position.x)
        self.ant.move(best_angle)
        self.ant.deposit_pheromone(self.pheromone_map)
        self.pheromone_map.evaporate()

        # Check for food collection
        for food in self.food_sources[:]:
            if self.ant.position.distance_to(food) < 2:
                self.food_sources.remove(food)
                print("Food collected!")

    def update_display(self):
        self.axes[0].clear()
        self.axes[0].set_title("Ant Path")
        self.axes[0].set_xlim(0, self.width)
        self.axes[0].set_ylim(0, self.height)
        x_vals = [p.x for p in self.ant.path]
        y_vals = [p.y for p in self.ant.path]
        self.axes[0].plot(x_vals, y_vals, 'b-')
        self.axes[0].plot(self.ant.position.x, self.ant.position.y, 'ro')

        self.axes[1].clear()
        self.axes[1].set_title("Pheromone Map")
        self.axes[1].imshow(self.pheromone_map.grid, cmap='hot', origin='lower', extent=[0, self.width, 0, self.height])

        self.axes[2].clear()
        self.axes[2].set_title("Ant Metrics")
        self.axes[2].text(0.1, 0.8, f"Speed: {self.ant.speed:.2f}", fontsize=12)
        self.axes[2].text(0.1, 0.6, f"Path Length: {len(self.ant.path)}", fontsize=12)
        self.axes[2].text(0.1, 0.4, f"Pheromone Total: {np.sum(self.pheromone_map.grid):.2f}", fontsize=12)
        self.axes[2].axis('off')

        plt.pause(0.01)

    def run(self):
        while self.running and self.food_sources:
            self.simulate_step()
            self.update_display()
            time.sleep(0.1)
        plt.show()

if __name__ == "__main__":
    sim = AntSLAMSingle()
    sim.run()
