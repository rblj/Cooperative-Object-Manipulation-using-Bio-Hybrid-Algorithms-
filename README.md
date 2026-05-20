# 🧠 Cooperative Object Manipulation using Bio-Hybrid Algorithms

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Webots](https://img.shields.io/badge/Simulation-Webots-orange)
![MAPPO](https://img.shields.io/badge/Reinforcement%20Learning-MAPPO-green)
![Research](https://img.shields.io/badge/Research-Thesis%20Project-red)

A bio-hybrid multi-agent robotic framework integrating Reinforcement Learning and Swarm Intelligence algorithms for cooperative object manipulation in constrained environments.

</div>

---

# 📌 Overview

This thesis presents a **Bio-Hybrid Multi-Agent Robotic Framework** designed for cooperative object manipulation using swarm-inspired intelligence and reinforcement learning.

The study integrates:

- 🐜 Ant Colony Optimization (ACO)
- 🐟 Particle Swarm Optimization (PSO)
- 🐦 Flocking Birds Algorithm
- 🧠 Multi-Agent Proximal Policy Optimization (MAPPO)

The framework enables multiple robots to coordinate and push rigid objects toward a target location using decentralized decision-making, local perception, and swarm-based coordination strategies.

The system was simulated and evaluated in WebotsR2025a Environment to analyze:

- Formation stability
- Coordination efficiency
- Cooperative pushing behavior
- Path consistency
- Swarm robustness

compared to the standard MAPPO baseline.

---

# 🎯 Research Objectives

- Improve cooperative object manipulation in multi-agent robotic systems
- Enhance coordination through bio-inspired swarm algorithms
- Compare Bio-Hybrid MAPPO performance against standard MAPPO
- Analyze formation stability and coordination efficiency during object pushing tasks

---

# 🤖 Robot Platform

## Simulation Robot: SUMMIT XL STEEL

<p align="center">
  <img src="images/robot.png" width="500">
</p>

The simulation utilizes the **SUMMIT XL STEEL** mobile robotic platform integrated within webots Environment.

The SUMMIT XL STEEL is a heavy-duty omnidirectional mobile robot designed for autonomous navigation, logistics, and robotic research applications. Its maneuverability and cooperative transport capabilities make it suitable for multi-agent object manipulation experiments.

---

## 🔧 Robot Specifications

- 4-Wheel Omnidirectional / Mecanum Drive
- Autonomous Navigation Support
- ROS-Compatible Architecture
- Real-Time Sensor Integration
- Differential & Swarm-Based Navigation
- Cooperative Object Pushing Capability

---

## 📡 Sensors & Components

The robot platform integrates multiple sensing and communication systems including:

- Distance Sensors
- Position Tracking
- Communication Signals
- IMU / Motion Tracking
- Collision Detection
- Cooperative Navigation Support

These components allow agents to:

- Detect nearby robots
- Maintain swarm formations
- Coordinate object pushing
- Navigate toward target goals
- Avoid inter-agent collisions

---

# 🌍 Simulation Environment

<p align="center">
  <img src="images/simulation_arena.png" width="850">
</p>

## Environment Description

The simulation environment consists of:

- Cooperative box-pushing arena
- Goal destination area
- Multi-agent navigation space
- Constrained pathways
- Swarm coordination testing zones

The environment was designed to evaluate the coordination capability of multiple autonomous agents under constrained object manipulation scenarios.

---

# 🎥 Early Training Trials

## Initial Simulation Comparisons

<table>
<tr>
<td align="center">

### 🐦 Flocking Birds Algorithm

<p align="center">
  <vid src="<img width="400" height="225" alt="Bio_Hybrid_Flocking" src="https://github.com/user-attachments/assets/21373bc8-b83e-4be3-894a-949f03721248" /><img width="400" height="225" alt="Bio_Hybrid_BeeSwarm" src="https://github.com/user-attachments/assets/ee8a7bdf-a4c4-4fa7-8e03-d0b6bdcfe58f" />

</p>

</td>

<td align="center">

### 🐝 Bee Swarm Algorithm

<p align="center">
  <vid src="<img width="400" height="225" alt="Bio_Hybrid_Flocking" src="https://github.com/user-attachments/assets/21373bc8-b83e-4be3-894a-949f03721248" /><img width="400" height="225" alt="Bio_Hybrid_BeeSwarm" src="https://github.com/user-attachments/assets/04176191-834b-48dc-9701-8922dff3622c" />

</p>

</td>
</tr>
</table>

---

# 📊 Training Results & Analysis

## Algorithm Comparison

<table>
<tr>
<td align="center">
<img src="images/flockingbirds_vs_beeswarm.png" width="400">
<br>
<b>Flocking Birds Algorithm VS Bee Swarm Algorithm</b>
</td>

<td align="center">
<img src="images/pure_vs_bio.png" width="400">
<br>
<b>Baseline MAPPO VS Bio-Hybrid MAPPO</b>
</td>
</tr>
</table>

---

# 🧬 Methodology

The framework follows a **Centralized Training Decentralized Execution (CTDE)** architecture where:

- Agents are trained collectively using MAPPO
- Execution occurs independently during deployment
- Swarm heuristics guide coordination behavior
- Local information and communication cues enable cooperation
- No predefined leader or role assignment exists

The hybrid framework combines reinforcement learning with biological coordination heuristics to improve cooperative transport behavior and reduce coordination failures.

---

# ⚙️ Technologies Used

## Simulation & Development

- WebotsR2025
- Python

## Reinforcement Learning

- MAPPO (Multi-Agent PPO)
- CTDE Framework

## Bio-Hybrid Algorithms

- Ant Colony Optimization (ACO)
- Particle Swarm Optimization (PSO)
- Flocking Birds Algorithm

## Monitoring & Visualization

- TensorBoard

---

# 🚀 Running the Project

## 1. Open Webots

- Launch Webots
- Load the simulation world
- Ensure the simulation is paused at:

```text
0:00:00:000
```

---

## 2. Open Command Prompt / Terminal

Navigate to the project directory(example):

```bash
cd "D:\4TH YEAR\THESIS\Cooperative-Object-Manipulation-using-Bio-Hybrid-Algorithm-main"
```

---

## 3. Run the Training Script

```bash
python train_swarm.py
```

Wait until the terminal displays:

```text
SWARM STANDBY
```

---

## 4. Start the Simulation

Return to Webots and press the **Play** button.

---

## 5. Launch TensorBoard

Open a second terminal window and run:

```bash
python -m tensorboard.main --logdir="ppo_swarm_logs"
```

---

## 6. Monitor Training Progress

Open the following in your browser:

```text
http://localhost:6006/#timeseries
```

TensorBoard can monitor:

- Reward progression
- Agent learning performance
- Swarm behavior trends
- Training stability
- Run comparisons

---

# 📈 Evaluation Metrics

The framework was evaluated using:

- ⏱️ Task Completion Time
- 🎯 Success Rate
- 📏 Path Efficiency
- 🤖 Formation Stability
- 📡 Inter-Agent Distance Variance
- 👥 Minimum Active Agents Required

---

# 🔬 Key Findings

- Bio-hybrid coordination improved formation consistency
- Swarm heuristics reduced coordination failures
- Flocking Birds produced smoother cooperative formations
- Hybrid MAPPO outperformed standard MAPPO in maintaining stable pushing behavior
- Swarm-based coordination improved cooperative transport efficiency

---

# 📚 Future Improvements

- Dynamic obstacle environments
- Real-world robot deployment
- Fault-tolerant swarm systems
- Adaptive communication mechanisms
- Vision-Language Model integration
- Real-time cooperative path planning

---

# 📂 Repository Structure

```text
├── controllers/
├── worlds/
├── images/
├── videos/
├── ppo_swarm_logs/
├── train_swarm.py
├── README.md
```

---

# 👨‍🔬 Researchers

Rubilee S. Ganoy and Nica Mae A. Cursat | Developed as a thesis project in Robotics, Artificial Intelligence, and Multi-Agent Systems research.

---

# 📜 Citation

```text
Cooperative Object Manipulation using Bio-Hybrid Algorithms
2026
```
