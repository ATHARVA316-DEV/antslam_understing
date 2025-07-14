import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import random
from collections import deque

# --- CONFIGURABLE PARAMETERS ---
GRID_SIZE = (20, 20, 20)
NUM_FOOD = 8
NUM_OBSTACLES = 80
PHEROMONE_DECAY = 0.01
PHEROMONE_DEPOSIT_SEARCH = 0.2
PHEROMONE_DEPOSIT_CARRY = 1.0
ANT_MEMORY = 12
MAX_STEPS = 1200

# --- DATA CLASSES ---
class Point3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    def as_tuple(self):
        return (self.x, self.y, self.z)
    def __eq__(self, other):
        return isinstance(other, Point3D) and self.x == other.x and self.y == other.y and self.z == other.z
    def __hash__(self):
        return hash((self.x, self.y, self.z))

# --- ENVIRONMENT ---
class Environment3D:
    EMPTY, OBSTACLE, FOOD, NEST = 0, 1, 2, 3
    def __init__(self, grid_size, num_food, num_obstacles):
        self.grid_size = grid_size
        self.grid = np.zeros(grid_size, dtype=int)
        self.nest = Point3D(grid_size[0] // 2, grid_size[1] // 2, grid_size[2] // 2)
        self.place_nest()
        self.food_points = self.place_random(num_food, self.FOOD)
        self.obstacle_points = self.place_random(num_obstacles, self.OBSTACLE)
    def place_nest(self):
        self.grid[self.nest.x, self.nest.y, self.nest.z] = self.NEST
    def place_random(self, count, kind):
        points = set()
        while len(points) < count:
            x = random.randint(0, self.grid_size[0]-1)
            y = random.randint(0, self.grid_size[1]-1)
            z = random.randint(0, self.grid_size[2]-1)
            if self.grid[x, y, z] == self.EMPTY:
                self.grid[x, y, z] = kind
                points.add(Point3D(x, y, z))
        return points
    def is_obstacle(self, pt):
        return self.grid[pt.x, pt.y, pt.z] == self.OBSTACLE
    def is_food(self, pt):
        return self.grid[pt.x, pt.y, pt.z] == self.FOOD
    def is_nest(self, pt):
        return self.grid[pt.x, pt.y, pt.z] == self.NEST
    def remove_food(self, pt):
        if self.is_food(pt):
            self.grid[pt.x, pt.y, pt.z] = self.EMPTY
            self.food_points.discard(pt)

# --- PHEROMONE MAP ---
class PheromoneMap3D:
    def __init__(self, grid_size):
        self.map = np.zeros(grid_size, dtype=float)
    def evaporate(self, decay):
        self.map *= (1.0 - decay)
    def deposit(self, pt, amount):
        self.map[pt.x, pt.y, pt.z] += amount
    def get(self, pt):
        return self.map[pt.x, pt.y, pt.z]

# --- ANT ---
class Ant3D:
    def __init__(self, env, pheromones, memory=12):
        self.env = env
        self.pheromones = pheromones
        self.pos = Point3D(env.nest.x, env.nest.y, env.nest.z)
        self.carrying_food = False
        self.path = [self.pos.as_tuple()]
        self.memory = deque(maxlen=memory)
        self.speed_history = []
        self.steps = 0
        self.last_direction = None
    def neighbors(self, pt):
        # 6 directions: +/-x, +/-y, +/-z
        dirs = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
        for dx, dy, dz in dirs:
            nx, ny, nz = pt.x + dx, pt.y + dy, pt.z + dz
            if 0 <= nx < self.env.grid_size[0] and 0 <= ny < self.env.grid_size[1] and 0 <= nz < self.env.grid_size[2]:
                yield Point3D(nx, ny, nz), (dx, dy, dz)
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
        if not options:
            for n, direction in self.neighbors(self.pos):
                if not self.env.is_obstacle(n):
                    pheromone = self.pheromones.get(n)
                    pheromone_levels.append(pheromone)
                    options.append(n)
                    directions.append(direction)
        if options:
            if self.carrying_food:
                nest = self.env.nest
                options_with_dirs = list(zip(options, directions))
                options_with_dirs.sort(key=lambda pd: abs(pd[0].x - nest.x) + abs(pd[0].y - nest.y) + abs(pd[0].z - nest.z))
                next_pos, next_dir = options_with_dirs[0]
            else:
                pheromone_arr = np.array(pheromone_levels) + 0.1
                probs = pheromone_arr / pheromone_arr.sum()
                idx = random.choices(range(len(options)), weights=probs, k=1)[0]
                next_pos = options[idx]
                next_dir = directions[idx]
        else:
            next_pos = self.pos
            next_dir = self.last_direction
        # Speed logic
        speed = 1.0
        if self.last_direction is not None and next_dir is not None:
            if next_dir == self.last_direction:
                speed = 1.0
            elif (next_dir[0] == -self.last_direction[0] and next_dir[1] == -self.last_direction[1] and next_dir[2] == -self.last_direction[2]):
                speed = 0.4
            else:
                speed = 0.7
        self.memory.append(self.pos)
        self.pos = next_pos
        self.path.append(self.pos.as_tuple())
        self.speed_history.append(speed)
        self.last_direction = next_dir
        if not self.carrying_food and self.env.is_food(self.pos):
            self.carrying_food = True
            self.env.remove_food(self.pos)
        elif self.carrying_food and self.env.is_nest(self.pos):
            self.carrying_food = False
        amount = PHEROMONE_DEPOSIT_CARRY if self.carrying_food else PHEROMONE_DEPOSIT_SEARCH
        self.pheromones.deposit(self.pos, amount)

# --- SIMULATION CONTROLLER ---
class SimulationController3D:
    def __init__(self):
        self.env = Environment3D(GRID_SIZE, NUM_FOOD, NUM_OBSTACLES)
        self.pheromones = PheromoneMap3D(GRID_SIZE)
        self.ant = Ant3D(self.env, self.pheromones, memory=ANT_MEMORY)
        self.steps = 0
        self.z_slice = GRID_SIZE[2] // 2
        self.fig = plt.figure(figsize=(14, 8))
        self.ax3d = self.fig.add_subplot(221, projection='3d')
        self.ax_pher = self.fig.add_subplot(222)
        self.ax_speed = self.fig.add_subplot(223)
        self.ax_env = self.fig.add_subplot(224)
        plt.subplots_adjust(left=0.1, bottom=0.18)
        axcolor = 'lightgoldenrodyellow'
        self.ax_slider = plt.axes((0.25, 0.05, 0.5, 0.03), facecolor=axcolor)
        self.slider = Slider(self.ax_slider, 'Z-slice', 0, GRID_SIZE[2]-1, valinit=self.z_slice, valstep=1)
        self.slider.on_changed(self.update_zslice)
    def update_zslice(self, val):
        self.z_slice = int(val)
        self.update_visuals()
    def update_visuals(self):
        # 3D Path
        self.ax3d.clear()
        self.ax3d.set_title('Ant Path (3D)')
        path = np.array(self.ant.path)
        self.ax3d.plot(path[:,0], path[:,1], path[:,2], color='blue')
        self.ax3d.scatter(*self.env.nest.as_tuple(), color='red', marker='*', s=80, label='Nest')
        for food in self.env.food_points:
            self.ax3d.scatter(food.x, food.y, food.z, color='green', marker='o', s=40)
        self.ax3d.scatter(self.ant.pos.x, self.ant.pos.y, self.ant.pos.z, color='orange', marker='o', s=60, label='Ant')
        self.ax3d.set_xlim(0, GRID_SIZE[0]-1)
        self.ax3d.set_ylim(0, GRID_SIZE[1]-1)
        self.ax3d.set_zlim(0, GRID_SIZE[2]-1)
        self.ax3d.legend()
        # Pheromone map (2D slice)
        self.ax_pher.clear()
        self.ax_pher.set_title(f'Pheromone Map (Z={self.z_slice})')
        pm_slice = self.pheromones.map[:,:,self.z_slice].T
        self.ax_pher.imshow(pm_slice, origin='lower', cmap='plasma', interpolation='nearest')
        # Speed graph
        self.ax_speed.clear()
        self.ax_speed.set_title('Ant Speed Over Time')
        self.ax_speed.plot(self.ant.speed_history, color='green')
        self.ax_speed.set_xlabel('Step')
        self.ax_speed.set_ylabel('Speed')
        # Environment map (2D slice)
        self.ax_env.clear()
        self.ax_env.set_title(f'Environment Map (Z={self.z_slice})')
        env_slice = self.env.grid[:,:,self.z_slice].T
        cmap = plt.cm.get_cmap('tab10', 5)
        self.ax_env.imshow(env_slice, origin='lower', cmap=cmap, interpolation='nearest')
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
                print('All food collected!')
                break
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    sim = SimulationController3D()
    sim.run() 