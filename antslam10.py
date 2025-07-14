import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
from collections import deque

# --- CONFIGURABLE PARAMETERS ---
GRID_SIZE = (30, 30)
NUM_FOOD = 5
NUM_OBSTACLES = 40
PHEROMONE_DECAY = 0.01
PHEROMONE_DEPOSIT_SEARCH = 0.2
PHEROMONE_DEPOSIT_CARRY = 1.0
ANT_MEMORY = 10  # Steps to avoid revisiting
MAX_STEPS = 1000

# --- DATA CLASSES ---
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def as_tuple(self):
        return (self.x, self.y)

    def __eq__(self, other):
        return isinstance(other, Point) and self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

# --- ENVIRONMENT ---
class Environment:
    EMPTY, OBSTACLE, FOOD, NEST = 0, 1, 2, 3

    def __init__(self, grid_size, num_food, num_obstacles):
        self.grid_size = grid_size
        self.grid = np.zeros(grid_size, dtype=int)
        self.nest = Point(grid_size[0] // 2, grid_size[1] // 2)
        self.place_nest()
        self.food_points = self.place_random(num_food, self.FOOD)
        self.obstacle_points = self.place_random(num_obstacles, self.OBSTACLE)

    def place_nest(self):
        self.grid[self.nest.x, self.nest.y] = self.NEST

    def place_random(self, count, kind):
        points = set()
        while len(points) < count:
            x, y = random.randint(0, self.grid_size[0]-1), random.randint(0, self.grid_size[1]-1)
            if self.grid[x, y] == self.EMPTY:
                self.grid[x, y] = kind
                points.add(Point(x, y))
        return points

    def is_obstacle(self, pt):
        return self.grid[pt.x, pt.y] == self.OBSTACLE

    def is_food(self, pt):
        return self.grid[pt.x, pt.y] == self.FOOD

    def is_nest(self, pt):
        return self.grid[pt.x, pt.y] == self.NEST

    def remove_food(self, pt):
        if self.is_food(pt):
            self.grid[pt.x, pt.y] = self.EMPTY
            self.food_points.discard(pt)

# --- PHEROMONE MAP ---
class PheromoneMap:
    def __init__(self, grid_size):
        self.map = np.zeros(grid_size, dtype=float)

    def evaporate(self, decay):
        self.map *= (1.0 - decay)

    def deposit(self, pt, amount):
        self.map[pt.x, pt.y] += amount

    def get(self, pt):
        return self.map[pt.x, pt.y]

# --- ANT ---
class Ant:
    def __init__(self, env, pheromones, memory=10):
        self.env = env
        self.pheromones = pheromones
        self.pos = Point(env.nest.x, env.nest.y)
        self.carrying_food = False
        self.path = [self.pos.as_tuple()]
        self.memory = deque(maxlen=memory)
        self.speed_history = []
        self.steps = 0
        # Track last direction (dx, dy); start with None
        self.last_direction = None

    def neighbors(self, pt):
        # Directions: up, down, left, right
        dirs = [(-1,0), (1,0), (0,-1), (0,1)]
        for dx, dy in dirs:
            nx, ny = pt.x + dx, pt.y + dy
            if 0 <= nx < self.env.grid_size[0] and 0 <= ny < self.env.grid_size[1]:
                yield Point(nx, ny), (dx, dy)

    def move(self):
        self.steps += 1
        options = []
        pheromone_levels = []
        directions = []
        for n, direction in self.neighbors(self.pos):
            if self.env.is_obstacle(n) or n in self.memory:
                continue
            pheromone = self.pheromones.get(n)
            pheromone_levels.append(pheromone)
            options.append(n)
            directions.append(direction)

        # If no options, allow revisiting memory
        if not options:
            for n, direction in self.neighbors(self.pos):
                if not self.env.is_obstacle(n):
                    pheromone = self.pheromones.get(n)
                    pheromone_levels.append(pheromone)
                    options.append(n)
                    directions.append(direction)

        # Exploration vs exploitation
        if options:
            if self.carrying_food:
                # Prefer nest direction (greedy)
                nest = self.env.nest
                options_with_dirs = list(zip(options, directions))
                options_with_dirs.sort(key=lambda pd: abs(pd[0].x - nest.x) + abs(pd[0].y - nest.y))
                next_pos, next_dir = options_with_dirs[0]
            else:
                # Weighted random by pheromone (plus small random for exploration)
                pheromone_arr = np.array(pheromone_levels) + 0.1  # Avoid zero
                probs = pheromone_arr / pheromone_arr.sum()
                idx = random.choices(range(len(options)), weights=probs, k=1)[0]
                next_pos = options[idx]
                next_dir = directions[idx]
        else:
            next_pos = self.pos  # Stuck
            next_dir = self.last_direction

        # Calculate speed based on turn
        speed = 1.0  # Default
        if self.last_direction is not None and next_dir is not None:
            # Compare last_direction and next_dir
            if next_dir == self.last_direction:
                speed = 1.0  # Straight
            elif (next_dir[0] == -self.last_direction[0] and next_dir[1] == -self.last_direction[1]):
                speed = 0.4  # Reverse
            else:
                speed = 0.7  # Left or right turn
        # If first move, just use 1.0

        # Update memory and path
        self.memory.append(self.pos)
        self.pos = next_pos
        self.path.append(self.pos.as_tuple())
        self.speed_history.append(speed)
        self.last_direction = next_dir

        # Interact with environment
        if not self.carrying_food and self.env.is_food(self.pos):
            self.carrying_food = True
            self.env.remove_food(self.pos)
        elif self.carrying_food and self.env.is_nest(self.pos):
            self.carrying_food = False

        # Deposit pheromone
        amount = PHEROMONE_DEPOSIT_CARRY if self.carrying_food else PHEROMONE_DEPOSIT_SEARCH
        self.pheromones.deposit(self.pos, amount)

# --- SIMULATION CONTROLLER ---
class SimulationController:
    def __init__(self):
        self.env = Environment(GRID_SIZE, NUM_FOOD, NUM_OBSTACLES)
        self.pheromones = PheromoneMap(GRID_SIZE)
        self.ant = Ant(self.env, self.pheromones, memory=ANT_MEMORY)
        self.steps = 0

        # Setup matplotlib
        self.fig = plt.figure(constrained_layout=True, figsize=(12, 8))
        self.gs = gridspec.GridSpec(2, 2, figure=self.fig)
        self.ax_pheromone = self.fig.add_subplot(self.gs[0, 0])
        self.ax_speed = self.fig.add_subplot(self.gs[0, 1])
        self.ax_path = self.fig.add_subplot(self.gs[1, 0])
        self.ax_env = self.fig.add_subplot(self.gs[1, 1])

    def update_visuals(self):
        self.ax_pheromone.clear()
        self.ax_pheromone.set_title("Pheromone Map")
        pm = self.pheromones.map
        self.ax_pheromone.imshow(pm.T, origin='lower', cmap='plasma', interpolation='nearest')
        self.ax_pheromone.plot(*zip(*self.ant.path), color='white', alpha=0.3, linewidth=1)

        self.ax_speed.clear()
        self.ax_speed.set_title("Ant Speed Over Time")
        self.ax_speed.plot(self.ant.speed_history, color='green')
        self.ax_speed.set_xlabel("Step")
        self.ax_speed.set_ylabel("Speed (cells/step)")

        self.ax_path.clear()
        self.ax_path.set_title("Ant Path")
        self.ax_path.imshow(self.env.grid.T, origin='lower', cmap='gray_r', alpha=0.3)
        self.ax_path.plot(*zip(*self.ant.path), color='blue', linewidth=2)
        self.ax_path.scatter(*self.env.nest.as_tuple(), color='red', marker='*', s=100, label='Nest')
        for food in self.env.food_points:
            self.ax_path.scatter(food.x, food.y, color='green', marker='o', s=60, label='Food')
        self.ax_path.legend(loc='upper right')

        self.ax_env.clear()
        self.ax_env.set_title("Environment Map")
        env_map = np.copy(self.env.grid)
        env_map[self.ant.pos.x, self.ant.pos.y] = 4  # Mark ant
        cmap = plt.cm.get_cmap('tab10', 5)
        self.ax_env.imshow(env_map.T, origin='lower', cmap=cmap, interpolation='nearest')
        self.ax_env.set_xticks([])
        self.ax_env.set_yticks([])

        plt.pause(0.001)

    def run(self, max_steps=MAX_STEPS):
        plt.ion()
        for _ in range(max_steps):
            self.ant.move()
            self.pheromones.evaporate(PHEROMONE_DECAY)
            self.update_visuals()
            if not self.env.food_points:
                print("All food collected!")
                break
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    sim = SimulationController()
    sim.run() 