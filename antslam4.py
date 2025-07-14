# Add this inside the existing AntSLAM class

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
import copy

@dataclass
class Point:
    x: float
    y: float
    
    def distance_to(self, other):
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

@dataclass
class Obstacle:
    x: float
    y: float
    width: float
    height: float
    
    def contains_point(self, point: Point) -> bool:
        return (self.x <= point.x <= self.x + self.width and 
                self.y <= point.y <= self.y + self.height)

class PheromoneMap:
    def __init__(self, width: int, height: int, grid_size: int = 50):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.pheromone_grid = np.zeros((grid_size, grid_size))
        self.evaporation_rate = 0.01
        
    def add_pheromone(self, x: float, y: float, strength: float = 1.0):
        """Add pheromone at given coordinates"""
        grid_x = int((x / self.width) * self.grid_size)
        grid_y = int((y / self.height) * self.grid_size)
        
        if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
            self.pheromone_grid[grid_y, grid_x] += strength
    
    def get_pheromone(self, x: float, y: float) -> float:
        """Get pheromone strength at given coordinates"""
        grid_x = int((x / self.width) * self.grid_size)
        grid_y = int((y / self.height) * self.grid_size)
        
        if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
            return self.pheromone_grid[grid_y, grid_x]
        return 0.0
    
    def evaporate(self):
        """Evaporate pheromones over time"""
        self.pheromone_grid *= (1 - self.evaporation_rate)
        self.pheromone_grid = np.maximum(self.pheromone_grid, 0)

class Ant:
    def __init__(self, x: float, y: float, ant_id: int):
        self.position = Point(x, y)
        self.ant_id = ant_id
        self.path = [Point(x, y)]
        self.angle = random.uniform(0, 2 * np.pi)
        self.speed = 2.0
        self.sensor_range = 20.0
        self.has_food = False
        self.home = Point(x, y)
        self.memory = []  # Store visited locations
        
    def move(self, pheromone_map: PheromoneMap, obstacles: List[Obstacle], 
             food_sources: List[Point], bounds: Tuple[float, float]):
        """Move ant based on pheromone trails and obstacles"""
        # Sense environment
        best_direction = self.sense_environment(pheromone_map, obstacles, food_sources)
        
        # Update angle with some randomness
        self.angle += random.uniform(-0.3, 0.3)
        if best_direction is not None:
            self.angle = 0.7 * self.angle + 0.3 * best_direction
        
        # Calculate new position
        new_x = self.position.x + self.speed * np.cos(self.angle)
        new_y = self.position.y + self.speed * np.sin(self.angle)
        
        # Check bounds
        new_x = max(10, min(bounds[0] - 10, new_x))
        new_y = max(10, min(bounds[1] - 10, new_y))
        
        new_position = Point(new_x, new_y)
        
        # Check for obstacles
        collision = False
        for obstacle in obstacles:
            if obstacle.contains_point(new_position):
                collision = True
                break
        
        if not collision:
            self.position = new_position
            self.path.append(copy.deepcopy(self.position))
            
            # Keep path length manageable
            if len(self.path) > 100:
                self.path.pop(0)
            
            # Add to memory
            self.memory.append(copy.deepcopy(self.position))
            if len(self.memory) > 50:
                self.memory.pop(0)
        else:
            # Change direction if collision
            self.angle += random.uniform(-np.pi/2, np.pi/2)
    
    def sense_environment(self, pheromone_map: PheromoneMap, obstacles: List[Obstacle], 
                         food_sources: List[Point]) -> Optional[float]:
        """Sense pheromones and food in the environment"""
        # Check for food
        if not self.has_food:
            for food in food_sources:
                if self.position.distance_to(food) < 15:
                    self.has_food = True
                    return np.arctan2(self.home.y - self.position.y, 
                                    self.home.x - self.position.x)
        
        # If carrying food, head home
        if self.has_food:
            if self.position.distance_to(self.home) < 10:
                self.has_food = False
            return np.arctan2(self.home.y - self.position.y, 
                            self.home.x - self.position.x)
        
        # Sense pheromones in multiple directions
        directions = np.linspace(0, 2*np.pi, 8)
        best_strength = 0
        best_direction = None
        
        for direction in directions:
            sense_x = self.position.x + self.sensor_range * np.cos(direction)
            sense_y = self.position.y + self.sensor_range * np.sin(direction)
            
            strength = pheromone_map.get_pheromone(sense_x, sense_y)
            if strength > best_strength:
                best_strength = strength
                best_direction = direction
        
        return best_direction
    
    def deposit_pheromone(self, pheromone_map: PheromoneMap):
        """Deposit pheromone at current position"""
        strength = 2.0 if self.has_food else 0.5
        pheromone_map.add_pheromone(self.position.x, self.position.y, strength)

class AntSLAM:
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.pheromone_map = PheromoneMap(width, height)
        self.ants = []
        self.obstacles = []
        self.food_sources = []
        self.nest_location = Point(width // 2, height // 2)
        
        # Create initial setup
        self.setup_environment()
        
        # Visualization
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.set_aspect('equal')
        self.ax.set_title('AntSLAM - Ant Colony Simultaneous Localization and Mapping')
        
        # Control variables
        self.running = False
        self.show_pheromones = True
        self.show_paths = True
        self.edit_mode = False
        
        # Setup UI
        self.setup_ui()
        
        # Animation
        self.animation = None
        
    def setup_environment(self):
        """Initialize the environment with ants, obstacles, and food"""
        # Create ants around the nest
        for i in range(20):
            angle = random.uniform(0, 2 * np.pi)
            radius = random.uniform(10, 30)
            x = self.nest_location.x + radius * np.cos(angle)
            y = self.nest_location.y + radius * np.sin(angle)
            self.ants.append(Ant(x, y, i))
        
        # Create obstacles
        self.obstacles = [
            Obstacle(150, 200, 100, 50),
            Obstacle(400, 300, 80, 120),
            Obstacle(600, 100, 60, 80),
            Obstacle(200, 450, 120, 40),
            Obstacle(500, 500, 90, 70)
        ]
        
        # Create food sources
        self.food_sources = [
            Point(100, 100),
            Point(700, 150),
            Point(150, 500),
            Point(650, 450),
            Point(300, 50)
        ]
    
    def setup_ui(self):
        """Setup user interface controls"""
        plt.subplots_adjust(bottom=0.2)
        
        # Buttons
        ax_start = plt.axes([0.1, 0.05, 0.1, 0.04])
        self.btn_start = Button(ax_start, 'Start/Stop')
        self.btn_start.on_clicked(self.toggle_simulation)
        
        ax_reset = plt.axes([0.25, 0.05, 0.1, 0.04])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self.reset_simulation)
        
        ax_pheromone = plt.axes([0.4, 0.05, 0.1, 0.04])
        self.btn_pheromone = Button(ax_pheromone, 'Pheromones')
        self.btn_pheromone.on_clicked(self.toggle_pheromones)
        
        ax_paths = plt.axes([0.55, 0.05, 0.1, 0.04])
        self.btn_paths = Button(ax_paths, 'Paths')
        self.btn_paths.on_clicked(self.toggle_paths)
        
        ax_edit = plt.axes([0.7, 0.05, 0.1, 0.04])
        self.btn_edit = Button(ax_edit, 'Edit Mode')
        self.btn_edit.on_clicked(self.toggle_edit_mode)
        
        # Sliders
        ax_speed = plt.axes([0.1, 0.12, 0.3, 0.02])
        self.slider_speed = Slider(ax_speed, 'Speed', 0.1, 5.0, valinit=1.0)
        
        ax_evaporation = plt.axes([0.5, 0.12, 0.3, 0.02])
        self.slider_evaporation = Slider(ax_evaporation, 'Evaporation', 0.001, 0.1, valinit=0.01)
        
        # Mouse events for editing
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
    
    def toggle_simulation(self, event):
        """Start/stop the simulation"""
        if self.running:
            self.running = False
            if self.animation:
                self.animation.event_source.stop()
        else:
            self.running = True
            self.animation = FuncAnimation(self.fig, self.update, interval=50, blit=False)
    
    def reset_simulation(self, event):
        """Reset the simulation"""
        self.running = False
        if self.animation:
            self.animation.event_source.stop()
        
        self.pheromone_map = PheromoneMap(self.width, self.height)
        self.ants.clear()
        self.setup_environment()
        self.update_display()
    
    def toggle_pheromones(self, event):
        """Toggle pheromone visualization"""
        self.show_pheromones = not self.show_pheromones
        self.update_display()
    
    def toggle_paths(self, event):
        """Toggle path visualization"""
        self.show_paths = not self.show_paths
        self.update_display()
    
    def toggle_edit_mode(self, event):
        """Toggle edit mode for adding/removing obstacles and food"""
        self.edit_mode = not self.edit_mode
        if self.edit_mode:
            self.btn_edit.label.set_text('Exit Edit')
            print("Edit Mode: Left click to add food, Right click to add obstacle, 'r' to remove nearest item")
        else:
            self.btn_edit.label.set_text('Edit Mode')
    
    def on_mouse_click(self, event):
        """Handle mouse clicks for editing"""
        if not self.edit_mode or event.inaxes != self.ax:
            return
        
        if event.button == 1:  # Left click - add food
            self.food_sources.append(Point(event.xdata, event.ydata))
            self.update_display()
        elif event.button == 3:  # Right click - add obstacle
            self.obstacles.append(Obstacle(event.xdata - 25, event.ydata - 25, 50, 50))
            self.update_display()
    
    def on_key_press(self, event):
        """Handle key presses for editing"""
        if not self.edit_mode:
            return
        
        if event.key == 'r':  # Remove nearest item
            if event.inaxes == self.ax:
                click_point = Point(event.xdata, event.ydata)
                
                # Find nearest food source
                nearest_food = None
                min_food_dist = float('inf')
                for food in self.food_sources:
                    dist = click_point.distance_to(food)
                    if dist < min_food_dist:
                        min_food_dist = dist
                        nearest_food = food
                
                # Find nearest obstacle
                nearest_obstacle = None
                min_obstacle_dist = float('inf')
                for obstacle in self.obstacles:
                    obstacle_center = Point(obstacle.x + obstacle.width/2, obstacle.y + obstacle.height/2)
                    dist = click_point.distance_to(obstacle_center)
                    if dist < min_obstacle_dist:
                        min_obstacle_dist = dist
                        nearest_obstacle = obstacle
                
                # Remove the nearest item
                if min_food_dist < min_obstacle_dist and nearest_food:
                    self.food_sources.remove(nearest_food)
                elif nearest_obstacle:
                    self.obstacles.remove(nearest_obstacle)
                
                self.update_display()
    
    def update(self, frame):
        """Update simulation step"""
        if not self.running:
            return
        
        # Update simulation parameters
        speed_multiplier = self.slider_speed.val
        self.pheromone_map.evaporation_rate = self.slider_evaporation.val
        
        # Update ants
        for ant in self.ants:
            ant.speed = 2.0 * speed_multiplier
            ant.move(self.pheromone_map, self.obstacles, self.food_sources, (self.width, self.height))
            ant.deposit_pheromone(self.pheromone_map)
        
        # Evaporate pheromones
        self.pheromone_map.evaporate()
        
        # Update display
        self.update_display()
    
    def update_display(self):
        """Update the visualization"""
        self.ax.clear()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect('equal')
        self.ax.set_title('AntSLAM - Ant Colony Simultaneous Localization and Mapping')
        
        # Draw pheromone map
        if self.show_pheromones:
            extent = [0, self.width, 0, self.height]
            self.ax.imshow(self.pheromone_map.pheromone_grid, extent=extent, 
                          origin='lower', alpha=0.6, cmap='Reds')
        
        # Draw obstacles
        for obstacle in self.obstacles:
            rect = patches.Rectangle((obstacle.x, obstacle.y), obstacle.width, obstacle.height,
                                   linewidth=2, edgecolor='black', facecolor='gray')
            self.ax.add_patch(rect)
        
        # Draw food sources
        for food in self.food_sources:
            self.ax.plot(food.x, food.y, 'go', markersize=8, label='Food' if food == self.food_sources[0] else "")
        
        # Draw nest
        self.ax.plot(self.nest_location.x, self.nest_location.y, 'bs', markersize=12, label='Nest')
        
        # Draw ants and their paths
        for ant in self.ants:
            # Draw ant path
            if self.show_paths and len(ant.path) > 1:
                path_x = [p.x for p in ant.path]
                path_y = [p.y for p in ant.path]
                self.ax.plot(path_x, path_y, 'b-', alpha=0.3, linewidth=0.5)
            
            # Draw ant
            color = 'red' if ant.has_food else 'blue'
            self.ax.plot(ant.position.x, ant.position.y, 'o', color=color, markersize=4)
        
        # Add legend
        self.ax.legend(loc='upper right')
        
        # Add info text
        info_text = f"Ants: {len(self.ants)} | Food sources: {len(self.food_sources)} | Obstacles: {len(self.obstacles)}"
        if self.edit_mode:
            info_text += " | EDIT MODE: Left=Add Food, Right=Add Obstacle, R=Remove"
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.draw()
    
    def run(self):
        """Run the visualization"""
        self.update_display()
        plt.show()

# Create and run the AntSLAM simulation
if __name__ == "__main__":
    print("AntSLAM Simulation with Interactive Visualization")
    print("Controls:")
    print("- Start/Stop: Begin/pause simulation")
    print("- Reset: Reset simulation to initial state")
    print("- Pheromones: Toggle pheromone trail visualization")
    print("- Paths: Toggle ant path visualization")
    print("- Edit Mode: Add/remove obstacles and food sources")
    print("- Speed slider: Control simulation speed")
    print("- Evaporation slider: Control pheromone evaporation rate")
    print("\nEdit Mode Controls:")
    print("- Left click: Add food source")
    print("- Right click: Add obstacle")
    print("- Press 'r': Remove nearest item to cursor")
    print("\nStarting simulation...")
    
    slam = AntSLAM()
    slam.run()
class AntSLAM:
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.pheromone_map = PheromoneMap(width, height)
        self.ants = []
        self.obstacles = []
        self.food_sources = []
        self.nest_location = Point(width // 2, height // 2)

        # Metrics tracking
        self.food_collected_history = []
        self.pheromone_level_history = []
        self.time_step = 0

        # Create initial setup
        self.setup_environment()

        # Visualization
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.set_aspect('equal')
        self.ax.set_title('AntSLAM - Ant Colony Simultaneous Localization and Mapping')

        # Stats window
        self.fig_stats, self.ax_stats = plt.subplots(figsize=(6, 4))
        self.ax_stats.set_title("AntSLAM Metrics")
        self.ax_stats.set_xlabel("Time Step")
        self.ax_stats.set_ylabel("Values")

        # Control variables
        self.running = False
        self.show_pheromones = True
        self.show_paths = True
        self.edit_mode = False

        # Setup UI
        self.setup_ui()

        # Animation
        self.animation = None

    def update(self, frame):
        if not self.running:
            return

        speed_multiplier = self.slider_speed.val
        self.pheromone_map.evaporation_rate = self.slider_evaporation.val

        for ant in self.ants:
            ant.speed = 2.0 * speed_multiplier
            ant.move(self.pheromone_map, self.obstacles, self.food_sources, (self.width, self.height))
            ant.deposit_pheromone(self.pheromone_map)

        self.pheromone_map.evaporate()
        self.update_display()

        # Track metrics
        food_carriers = sum(1 for ant in self.ants if ant.has_food)
        total_pheromones = np.sum(self.pheromone_map.pheromone_grid)
        self.food_collected_history.append(food_carriers)
        self.pheromone_level_history.append(total_pheromones)
        self.time_step += 1

        # Update stats plot
        self.ax_stats.clear()
        self.ax_stats.set_title("AntSLAM Metrics")
        self.ax_stats.set_xlabel("Time Step")
        self.ax_stats.plot(self.food_collected_history, label="Food Carriers", color='green')
        self.ax_stats.plot(self.pheromone_level_history, label="Total Pheromone", color='red')
        self.ax_stats.legend()
        self.fig_stats.canvas.draw()

    def run(self):
        self.update_display()
        plt.show(block=False)  # Show main window non-blocking
        self.fig_stats.show()

    def save_metrics_to_csv(self, filename="ant_slam_data.csv"):
        df = pd.DataFrame({
            "Time Step": range(len(self.food_collected_history)),
            "Food Carriers": self.food_collected_history,
            "Total Pheromone": self.pheromone_level_history
        })
        df.to_csv(filename, index=False)
