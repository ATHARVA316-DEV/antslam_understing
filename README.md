# antslam_understing
This is my personal code developed for a better understanding of the AntSLAM algorithm. I'm using it to explore how bio-inspired navigation models work, and it’s mainly for experimental and learning purposes.


# 🐜 AntSLAM: Bioinspired Ant SLAM Simulation

A clean, modular, and well-visualized simulation of a single ant exploring a 2D or 3D environment, inspired by Ant Colony Optimization (ACO) and Simultaneous Localization and Mapping (SLAM) principles.

## Features
- **Single Ant Explorer**: Simulates a single ant navigating, foraging, and learning via pheromones.
- **2D and 3D Environments**: Choose between 2D (`antslam_simulation.py`) and 3D (`antslam_simulation_3d.py`) worlds.
- **Pheromone-based Learning**: Ant deposits and follows pheromone trails, balancing exploration and exploitation.
- **Obstacles & Food**: Randomly placed obstacles and food sources; ant returns food to the nest.
- **Real-time Visualization**: 
  - Pheromone map (heatmap or slice)
  - Ant speed over time
  - Path traveled
  - Environment map (with food, nest, obstacles, and ant)
- **Customizable Parameters**: Grid size, food/obstacle count, pheromone decay, etc.
- **Bioinspired Behavior**: Local decision-making, memory, and emergent path optimization.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <repo-directory>
   ```
2. **Install dependencies:**
   ```bash
   pip install numpy matplotlib
   ```

## Usage

### 2D Simulation
```bash
python antslam_simulation.py
```

### 3D Simulation
```bash
python antslam_simulation_3d.py
```

- The 3D version includes a slider to select the Z-slice for 2D map views.
- All visualizations update in real time.

## File Structure
- `antslam_simulation.py` — 2D AntSLAM simulation
- `antslam_simulation_3d.py` — 3D AntSLAM simulation
- `README.md` — Project documentation

## Customization
You can adjust simulation parameters (grid size, number of food/obstacles, pheromone decay, etc.) at the top of each script.

## Screenshots
_Add screenshots here after running the simulation!_

## Credits
- Developed by [Your Name]
- Inspired by natural ant behavior, ACO, and SLAM research

## License
MIT License (or your preferred license)
