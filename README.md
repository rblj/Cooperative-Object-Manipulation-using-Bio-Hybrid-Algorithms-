# Cooperative-Object-Manipulation-using-Bio-Hybrid-Algorithms-

A bio-hybrid multi-agent robotic framework combining reinforcement learning with swarm intelligence algorithms (ACO, PSO, and Flocking Birds) for cooperative object manipulation. Simulated in Webots, the system improved coordination, formation stability, and box-pushing efficiency over standard MAPPO.

## Running the Project

### 1. Open Webots
- Load the simulation world.
- Ensure that the simulation is **Paused** at:0:00:00:000

### 2. Open Command Prompt / Terminal

Navigate to the project directory:

cd "D:\4TH YEAR\THESIS"

Example:

cd "D:\4TH YEAR\THESIS\Cooperative-Object-Manipulation-using-Bio-Hybrid-Algorithm-main"

### 3. Run the Training Script

Execute the training environment:

python train_swarm.py

Wait until the terminal displays:

SWARM STANDBY

### 4. Start the Simulation

Return to Webots and press the Play button to begin the simulation.

### 5. Launch TensorBoard

Open a second Command Prompt / Terminal window and run:

python -m tensorboard.main --logdir="ppo_swarm_logs"

### 6. Monitor Training Progress

Open the following link in your browser:

http://localhost:6006/#timeseries

### TensorBoard can be used to monitor:
- Reward progression
- Agent learning performance
- Training trends
- Run comparisons
- Swarm behavior metrics

### Technologies Used
- Webots
- Python
- Reinforcement Learning (MAPPO)

### Bio-Hybrid Algorithms used in the Study:
- Ant Colony Optimization (ACO)
- Particle Swarm Optimization (PSO)
- Flocking Birds Algorithm

### Project Description

This project presents a bio-hybrid multi-agent robotic framework for cooperative object manipulation. The system combines reinforcement learning with swarm intelligence algorithms such as ACO, PSO, and Flocking Birds to improve coordination, formation stability, and object-pushing efficiency in simulated environments.
```
