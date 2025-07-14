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
python antslam10.py
```

### 3D Simulation
```bash
python antslam11.py
```

- The 3D version includes a slider to select the Z-slice for 2D map views.
- All visualizations update in real time.



## Customization
You can adjust simulation parameters (grid size, number of food/obstacles, pheromone decay, etc.) at the top of each script.

## Screenshots
antslam9
<img width="1913" height="1078" alt="Screenshot 2025-07-14 212557" src="https://github.com/user-attachments/assets/6eddfbc0-fa21-4abe-b471-41a7a86cb29b" />
antslam10
<img width="1919" height="1079" alt="Screenshot 2025-07-14 213026" src="https://github.com/user-attachments/assets/9dafab3b-a67a-4ac5-882a-e619e008d37e" />
antslam11
<img width="1919" height="1071" alt="Screenshot 2025-07-14 213717" src="https://github.com/user-attachments/assets/dd9af37d-bfea-44f4-90b2-46f4465d167b" />


## Credits
- Developed by Atharva M
- Inspired by natural ant behavior, ACO, and SLAM research

## License
MIT License (or your preferred license)
