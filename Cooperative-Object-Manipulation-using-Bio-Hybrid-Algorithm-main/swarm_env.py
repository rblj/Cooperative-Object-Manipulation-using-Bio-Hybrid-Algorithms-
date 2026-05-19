import gymnasium as gym
from gymnasium import spaces
import socket
import json
import numpy as np
import time
import select
from collections import deque
from typing import Dict, List, Tuple, Optional
import math

class SwarmEnvironment(gym.Env):
    """
    Cooperative Swarm Box Pushing Environment for MAPPO
    WEBOTS-ONLY - No internal simulation!
    
    INTEGRATED BIO-HYBRID ALGORITHMS:
    1. 🐜 ANT COLONY - Pheromone trails to guide agents to box
    2. 🐝 PARTICLE SWARM - Share best pushing positions
    3. 🦅 FLOCKING (BOIDS) - Maintain formation while pushing
    """
    
    def __init__(self, num_agents=5, port=12345, use_lidar=True, 
                 bio_algorithm='ant_colony', max_steps=300):
        super().__init__()
        self.num_agents = num_agents
        self.agent_ids = [f"robot{i+1}" for i in range(num_agents)]
        self.use_lidar = use_lidar
        self.lidar_sectors = 8
        self.bio_algorithm = bio_algorithm
        self.max_steps = max_steps
        self.num_roles = 3  # CHANGED FROM 5 TO 3!
        
        # Task parameters
        self.box_start_x = 12.0
        self.box_start_y = 0.0
        self.goal_x = -12.0
        self.goal_y = 0.0
        self.goal_radius = 2.0
        
        # ============= NO INTERNAL PHYSICS =============
        
        # === OBSERVATION SPACES ===
        if use_lidar:
            obs_dim = 4 + 4 + 2 + 2 + self.lidar_sectors + 3  # CHANGED FROM +5 TO +3
        else:
            obs_dim = 4 + 4 + 2 + 2 + 3  # CHANGED FROM +5 TO +3
            
        self.observation_space = spaces.Box(
            low=-50.0,
            high=50.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # === ACTION SPACES ===
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
        
        # === CENTRALIZED STATE SPACE ===
        state_dim = num_agents * 2 + 4 + 2 + 20  # CHANGED FROM 30 TO 20 (3 algorithms * 4 metrics + base)
        self.state_space = spaces.Box(
            low=-50.0,
            high=50.0,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # ============= SOCKET SERVER =============
        self.port = port
        self.server = None
        self.connections = {}
        self.robot_states = {}
        self.connection_attempts = {}
        
        # ============= EPISODE TRACKING =============
        self.step_count = 0
        self.episode_count = 0
        self.episode_reward = 0.0
        self.episode_collisions = 0
        self.episode_cooperations = 0
        
        # ============= 🐜 ANT COLONY MEMORY =============
        self.pheromone_grid = np.zeros((100, 100))
        self.pheromone_decay = 0.95
        self.pheromone_deposit = 1.0
        self.pheromone_trail_deposit = 0.2
        self.pheromone_evaporation = 0.05
        
        # ============= 🐝 PARTICLE SWARM MEMORY =============
        self.global_best_position = np.array([12.0, 0.0])
        self.global_best_reward = -float('inf')
        self.personal_best_positions = {}
        self.personal_best_rewards = {}
        self.pso_velocity = {}
        self.pso_inertia = 0.7
        self.pso_cognitive = 1.5
        self.pso_social = 1.5
        
        # ============= 🦅 FLOCKING (BOIDS) MEMORY =============
        self.flock_center = np.array([0.0, 0.0])
        self.flock_velocity = np.array([0.0, 0.0])
        self.cohesion_weight = 0.5
        self.alignment_weight = 0.5
        self.separation_weight = 0.8
        self.separation_distance = 1.5
        
        # ============= SHARED STATE =============
        self.box_position = np.array([self.box_start_x, self.box_start_y], dtype=np.float32)
        self.box_velocity = np.zeros(2, dtype=np.float32)
        self.robot_positions = {}
        self.robot_dist_to_box = {}
        self.visited_grid = np.zeros((100, 100))
        self.visited_decay = 0.99
        
        # Initialize server with retry
        self._init_server_with_retry()
        
        print("=" * 70)
        print("[Env] 🧬 WEBOTS-ONLY MODE - NO INTERNAL SIMULATION!")
        print("[Env] MAPPO Environment with 3 BIO-HYBRID ALGORITHMS")
        print("=" * 70)
        print("[Env] 🐜 Ant Colony     - Pheromone trails")
        print("[Env] 🐝 Particle Swarm - Share best positions")
        print("[Env] 🦅 Flocking       - Formation control")
        print("=" * 70)
        print(f"[Env] Current algorithm: {bio_algorithm.upper()}")
        print(f"[Env] Task: Move box from ({self.box_start_x},0) to ({self.goal_x},0)")
        print(f"[Env] Observation dim: {obs_dim}, Action dim: 2, State dim: {state_dim}")
        print(f"[Env] Role-based rewards enabled: {self.num_roles} roles")
        print(f"[Env] Waiting for {num_agents} robots to connect on port {port}...")
        print("=" * 70)
    
    def _init_server_with_retry(self, max_retries=5):
        """Initialize socket server with retry mechanism"""
        for attempt in range(max_retries):
            try:
                self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.server.bind(("localhost", self.port))
                self.server.listen(self.num_agents)
                self.server.setblocking(False)
                print(f"[Env] Server started on port {self.port}")
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[Env] Failed to start server (attempt {attempt+1}/{max_retries}): {e}")
                    time.sleep(1)
                    self.port += 1
                else:
                    print(f"[Env] Failed to start server after {max_retries} attempts: {e}")
                    raise
    
    # ============= 🐜 ANT COLONY METHODS =============
    def _update_ant_colony(self):
        """Update pheromone trails"""
        if self.bio_algorithm != 'ant_colony' and self.bio_algorithm != 'all':
            return
            
        grid_x = int((self.box_position[0] + 20) * 2)
        grid_y = int((self.box_position[1] + 20) * 2)
        grid_x = np.clip(grid_x, 0, 99)
        grid_y = np.clip(grid_y, 0, 99)
        self.pheromone_grid[grid_x, grid_y] += self.pheromone_deposit
        
        for robot_id in self.agent_ids:
            if robot_id in self.robot_states:
                pos = self.robot_states[robot_id].get('position', [0, 0])
                grid_x = int((pos[0] + 20) * 2)
                grid_y = int((pos[1] + 20) * 2)
                grid_x = np.clip(grid_x, 0, 99)
                grid_y = np.clip(grid_y, 0, 99)
                self.pheromone_grid[grid_x, grid_y] += self.pheromone_trail_deposit
        
        self.pheromone_grid *= (1 - self.pheromone_evaporation)
        self.pheromone_grid = np.clip(self.pheromone_grid, 0, 100)
    
    def _get_pheromone_gradient(self, position):
        """Get pheromone gradient for navigation"""
        x, y = position
        grid_x = int((x + 20) * 2)
        grid_y = int((y + 20) * 2)
        grid_x = np.clip(grid_x, 0, 99)
        grid_y = np.clip(grid_y, 0, 99)
        
        gradient = [0.0, 0.0, 0.0]
        
        if grid_x < 99:
            gradient[0] = self.pheromone_grid[grid_x + 1, grid_y]
        if grid_y > 0:
            gradient[1] = self.pheromone_grid[grid_x, grid_y - 1]
        if grid_y < 99:
            gradient[2] = self.pheromone_grid[grid_x, grid_y + 1]
        
        return gradient
    
    # ============= 🐝 PARTICLE SWARM METHODS =============
    def _update_particle_swarm(self):
        """Update PSO global best position"""
        if self.bio_algorithm != 'particle_swarm' and self.bio_algorithm != 'all':
            return
            
        current_reward = -np.linalg.norm(self.box_position - np.array([self.goal_x, self.goal_y]))
        
        if current_reward > self.global_best_reward:
            self.global_best_reward = current_reward
            self.global_best_position = self.box_position.copy()
        
        for robot_id in self.agent_ids:
            if robot_id in self.robot_states:
                pos = self.robot_states[robot_id].get('position', [0, 0])
                dist_to_box = np.linalg.norm(np.array(pos) - self.box_position)
                reward = -dist_to_box
                
                if robot_id not in self.personal_best_rewards or reward > self.personal_best_rewards[robot_id]:
                    self.personal_best_rewards[robot_id] = reward
                    self.personal_best_positions[robot_id] = np.array(pos)
                
                if robot_id not in self.pso_velocity:
                    self.pso_velocity[robot_id] = np.zeros(2)
    
    # ============= 🦅 FLOCKING (BOIDS) METHODS =============
    def _update_flocking(self):
        """Update flock center and velocity"""
        if self.bio_algorithm != 'flocking' and self.bio_algorithm != 'all':
            return
            
        positions = []
        velocities = []
        
        for robot_id in self.agent_ids:
            if robot_id in self.robot_states:
                pos = self.robot_states[robot_id].get('position', [0, 0])
                vel = self.robot_states[robot_id].get('velocity', [0, 0])
                positions.append(np.array(pos))
                velocities.append(np.array(vel))
        
        if positions:
            self.flock_center = np.mean(positions, axis=0)
        if velocities:
            self.flock_velocity = np.mean(velocities, axis=0)
    
    # ============= BIO-HYBRID SIGNAL COLLECTION =============
    def _get_bio_signals(self, robot_id, position):
        """Get all 3 bio-hybrid signals for robot observation"""
        signals = []
        
        # 1. 🐜 Ant Colony - Pheromone strength
        grid_x = int((position[0] + 20) * 2)
        grid_y = int((position[1] + 20) * 2)
        grid_x = np.clip(grid_x, 0, 99)
        grid_y = np.clip(grid_y, 0, 99)
        signals.append(float(self.pheromone_grid[grid_x, grid_y]))
        
        # 2. 🐝 Particle Swarm - Distance to global best
        dist_to_global_best = np.linalg.norm(position - self.global_best_position)
        signals.append(float(1.0 / (dist_to_global_best + 1.0)))
        
        # 3. 🦅 Flocking - Distance to flock center
        if np.any(self.flock_center):
            dist_to_flock = np.linalg.norm(position - self.flock_center)
            signals.append(float(1.0 / (dist_to_flock + 1.0)))
        else:
            signals.append(0.0)
        
        return signals
    
    # ============= ROLE-BASED REWARDS FOR 3 ALGORITHMS =============
    def _calculate_ant_colony_reward(self, robot_id, role_weight):
        """🐜 Ant Colony role reward"""
        if role_weight <= 0.05:
            return 0.0
        
        if robot_id in self.robot_states:
            pos = self.robot_states[robot_id].get('position', [0, 0])
            grid_x = int((pos[0] + 20) * 2)
            grid_y = int((pos[1] + 20) * 2)
            grid_x = np.clip(grid_x, 0, 99)
            grid_y = np.clip(grid_y, 0, 99)
            
            pheromone = self.pheromone_grid[grid_x, grid_y]
            return float(pheromone * role_weight * 0.3)
        
        return 0.0
    
    def _calculate_particle_swarm_reward(self, robot_id, role_weight):
        """🐝 Particle Swarm role reward"""
        if role_weight <= 0.05:
            return 0.0
        
        if robot_id in self.robot_states:
            pos = self.robot_states[robot_id].get('position', [0, 0])
            pos_np = np.array(pos)
            
            dist_to_best = np.linalg.norm(pos_np - self.global_best_position)
            reward = (5.0 / (dist_to_best + 1.0)) * role_weight
            
            if robot_id in self.personal_best_positions:
                dist_to_personal = np.linalg.norm(
                    pos_np - self.personal_best_positions[robot_id]
                )
                if dist_to_personal < 1.0:
                    reward += 1.0 * role_weight
            
            return float(reward)
        
        return 0.0
    
    def _calculate_flocking_reward(self, robot_id, role_weight):
        """🦅 Flocking role reward"""
        if role_weight <= 0.05:
            return 0.0
        
        if robot_id in self.robot_states and np.any(self.flock_center):
            pos = self.robot_states[robot_id].get('position', [0, 0])
            pos_np = np.array(pos)
            
            dist_to_center = np.linalg.norm(pos_np - self.flock_center)
            cohesion = (3.0 / (dist_to_center + 1.0)) * role_weight * 0.3
            
            separation = 0.0
            for other_id in self.agent_ids:
                if other_id != robot_id and other_id in self.robot_states:
                    other_pos = self.robot_states[other_id].get('position', [0, 0])
                    dist = np.linalg.norm(np.array(other_pos) - pos_np)
                    if dist < self.separation_distance:
                        separation -= (self.separation_distance - dist) * role_weight * 0.3
            
            return float(cohesion + separation)
        
        return 0.0
    
        # ============= MAIN REWARD FUNCTION =============
    def _calculate_reward(self, role_weights=None):
        """
        COMPREHENSIVE REWARD SYSTEM:
        - 💰 Box movement rewards
        - 🤝 Cooperation rewards  
        - 📦 Pushing rewards
        - 🚧 Collision penalties
        """
        reward = 0.0
        
        # ============= GET STATE =============
        box_x = float(self.box_position[0])
        box_vel_x = float(self.box_velocity[0])
        box_vel_magnitude = float(np.linalg.norm(self.box_velocity))
        
        # Track previous box position for progress calculation
        if not hasattr(self, 'prev_box_x'):
            self.prev_box_x = box_x
        
        # ============= 1. 🎯 TASK REWARD - Box moved toward goal =============
        progress = (self.prev_box_x - box_x)
        if progress > 0:
            progress_reward = progress * 20.0
            reward += progress_reward
            print(f"[Env] 📦 Box moved toward goal! Reward: +{progress_reward:.2f}")
        elif progress < 0:
            reward += progress * 5.0
            print(f"[Env] ⚠️ Box moved away from goal! Penalty: {progress*5:.2f}")
        
        if box_vel_x < 0:
            velocity_reward = abs(box_vel_x) * 10.0
            reward += velocity_reward
            print(f"[Env] 💨 Fast push! +{velocity_reward:.2f}")
        
        # ============= 2. 📦 BOX TOUCHING/ PUSHING REWARD =============
        touching_box_count = 0
        pushing_hard_count = 0
        
        for robot_id in self.agent_ids:
            if robot_id in self.robot_states:
                pos = self.robot_states[robot_id].get('position', [0, 0])
                vel = self.robot_states[robot_id].get('velocity', [0, 0])
                
                dist = np.linalg.norm(np.array(pos) - self.box_position)
                
                if dist < 1.0:
                    touching_box_count += 1
                    reward += 0.5
                    
                    if vel[0] > 0.1:
                        pushing_hard_count += 1
                        reward += 1.0
        
        # ============= 3. 🤝 COOPERATION REWARD =============
        if touching_box_count >= 2:
            cooperation_bonus = (touching_box_count ** 2) * 2.0
            reward += cooperation_bonus
            self.episode_cooperations += 1
            print(f"[Env] 🤝 TEAM PUSH! {touching_box_count} robots - Bonus: +{cooperation_bonus:.2f}")
        
        # ============= 4. 🧬 ROLE-BASED REWARDS =============
        if role_weights is not None:
            for i, robot_id in enumerate(self.agent_ids):
                if robot_id in self.robot_states and i < len(role_weights):
                    robot_role_weights = role_weights[i]
                    pos = self.robot_states[robot_id].get('position', [0, 0])
                    
                    # 🐜 ANT COLONY REWARD
                    if len(robot_role_weights) >= 1:
                        ant_weight = robot_role_weights[0]
                        if ant_weight > 0.1:
                            grid_x = int((pos[0] + 20) * 2)
                            grid_y = int((pos[1] + 20) * 2)
                            grid_x = np.clip(grid_x, 0, 99)
                            grid_y = np.clip(grid_y, 0, 99)
                            
                            pheromone = self.pheromone_grid[grid_x, grid_y]
                            if pheromone > 0:
                                ant_reward = pheromone * ant_weight * 0.2
                                reward += ant_reward
                    
                    # 🐝 PARTICLE SWARM REWARD
                    if len(robot_role_weights) >= 2:
                        pso_weight = robot_role_weights[1]
                        if pso_weight > 0.1:
                            dist_to_box = np.linalg.norm(np.array(pos) - self.box_position)
                            pso_reward = (5.0 / (dist_to_box + 1.0)) * pso_weight * 0.5
                            reward += pso_reward
                    
                    # 🦅 FLOCKING REWARD - FIXED!
                    if len(robot_role_weights) >= 3:
                        flock_weight = robot_role_weights[2]
                        if flock_weight > 0.1 and np.any(self.flock_center):
                            dist_to_flock = np.linalg.norm(np.array(pos) - self.flock_center)
                            flock_reward = (3.0 / (dist_to_flock + 1.0)) * flock_weight * 0.3
                            reward += flock_reward
        
        # ============= 5. 🚧 COLLISION PENALTIES =============
        collisions = 0
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                robot_i = self.agent_ids[i]
                robot_j = self.agent_ids[j]
                
                if robot_i in self.robot_states and robot_j in self.robot_states:
                    pos_i = np.array(self.robot_states[robot_i].get('position', [0, 0]))
                    pos_j = np.array(self.robot_states[robot_j].get('position', [0, 0]))
                    dist = np.linalg.norm(pos_i - pos_j)
                    
                    if dist < 0.8:
                        reward -= 10.0
                        collisions += 1
                        print(f"[Env] 💥 Robot collision! Penalty: -10.0")
        
        self.episode_collisions += collisions
        
        # ============= 6. ⏱️ TIME PENALTY =============
        reward -= 0.1
        
        # ============= 7. 🏆 SUCCESS BONUS =============
        distance_to_goal = np.linalg.norm(self.box_position - [self.goal_x, self.goal_y])
        if distance_to_goal < self.goal_radius:
            reward += 500.0
            print(f"[Env] 🏆 SUCCESS! Box reached goal! +500.0")
        
        self.prev_box_x = box_x
        
        return float(reward)
    
    def _get_observations(self):
        """Get decentralized observations - ENSURE BOX IS VISIBLE!"""
        observations = []
        
        for robot_id in self.agent_ids:
            if robot_id in self.robot_states:
                state = self.robot_states[robot_id]
                
                obs = []
                
                # 1. Own state (4)
                pos = state.get('position', [0, 0])
                vel = state.get('velocity', [0, 0])
                obs.extend([float(pos[0]), float(pos[1]), float(vel[0]), float(vel[1])])
                
                # 2. Box state (4) - CRITICAL: This is the BOX POSITION!
                box_pos = state.get('box_position', [self.box_start_x, self.box_start_y])
                box_vel = state.get('box_velocity', [0, 0])
                obs.extend([float(box_pos[0]), float(box_pos[1]), float(box_vel[0]), float(box_vel[1])])
                
                # 3. Goal position (2)
                obs.extend([float(self.goal_x), float(self.goal_y)])
                
                # 4. Relative position to box (2) - CRITICAL: Vector to box!
                obs.extend([float(box_pos[0] - pos[0]), float(box_pos[1] - pos[1])])
                
                # 5. LiDAR data (8) - SHOULD SHOW BOX!
                if self.use_lidar:
                    lidar = state.get('lidar', [10.0] * self.lidar_sectors)
                    obs.extend([float(x) for x in lidar[:self.lidar_sectors]])
                
                # 6. Bio signals
                bio_signals = self._get_bio_signals(robot_id, np.array(pos, dtype=np.float32))
                obs.extend(bio_signals)
                
                observations.append(np.array(obs, dtype=np.float32))
            else:
                obs_dim = self.observation_space.shape[0]
                observations.append(np.zeros(obs_dim, dtype=np.float32))
        
        return np.array(observations, dtype=np.float32)
    
    def _get_centralized_state(self):
        """
        Get centralized state with 3 bio metrics
        CRITICAL FIX: All operations use numpy arrays, never lists!
        """
        state = []
        
        # 1. All robot positions
        for robot_id in self.agent_ids:
            if robot_id in self.robot_states:
                pos = self.robot_states[robot_id].get('position', [0, 0])
                state.extend([float(pos[0]), float(pos[1])])
            else:
                state.extend([0.0, 0.0])
        
        # 2. Box state (4)
        state.extend([
            float(self.box_position[0]),
            float(self.box_position[1]),
            float(self.box_velocity[0]),
            float(self.box_velocity[1])
        ])
        
        # 3. Goal position (2)
        state.extend([float(self.goal_x), float(self.goal_y)])
        
        # 4. Cooperation metrics
        touching_box = 0
        total_distance = 0.0
        connected_robots = len(self.connections)
        
        for robot_id in self.agent_ids:
            if robot_id in self.robot_states:
                pos = self.robot_states[robot_id].get('position', [0, 0])
                pos_np = np.array([float(pos[0]), float(pos[1])])
                box_np = np.array([float(self.box_position[0]), float(self.box_position[1])])
                dist = float(np.linalg.norm(pos_np - box_np))
                total_distance += dist
                if dist < 2.0:
                    touching_box += 1
        
        avg_distance = total_distance / max(connected_robots, 1)
        
        box_np = np.array([float(self.box_position[0]), float(self.box_position[1])])
        goal_np = np.array([float(self.goal_x), float(self.goal_y)])
        distance_to_goal = float(np.linalg.norm(box_np - goal_np))
        
        connection_ratio = connected_robots / self.num_agents
        cooperation_rate = self.episode_cooperations / max(self.step_count, 1)
        
        # 5. ALL 3 BIO-HYBRID METRICS (4 per algorithm = 12 metrics)
        metrics = [
            # Base metrics (5)
            float(touching_box),
            float(avg_distance),
            float(distance_to_goal),
            float(connection_ratio),
            float(cooperation_rate),
            
            # 🐜 Ant Colony metrics (4)
            float(np.max(self.pheromone_grid)),
            float(np.mean(self.pheromone_grid)),
            float(np.std(self.pheromone_grid)),
            float(np.sum(self.pheromone_grid > 1)),
            
            # 🐝 Particle Swarm metrics (4)
            float(self.global_best_reward),
            float(np.linalg.norm(box_np - self.global_best_position)),
            float(len(self.personal_best_positions)),
            float(np.mean([np.linalg.norm(self.pso_velocity.get(rid, [0,0])) for rid in self.agent_ids if rid in self.pso_velocity]) if self.pso_velocity else 0.0),
            
            # 🦅 Flocking metrics (4)
            float(np.linalg.norm(self.flock_center)) if np.any(self.flock_center) else 0.0,
            float(np.linalg.norm(self.flock_velocity)) if np.any(self.flock_velocity) else 0.0,
            float(self.cohesion_weight),
            float(self.separation_weight),
            
            # Progress (1)
            float(self.step_count / max(self.max_steps, 1))
        ]
        state.extend(metrics)
        
        state_array = np.array(state, dtype=np.float32)
        
        expected_dim = self.num_agents * 2 + 4 + 2 + 20  # UPDATED: 5 base + 12 bio + 1 progress = 18? Actually 20
        if state_array.shape[0] != expected_dim:
            if state_array.shape[0] > expected_dim:
                state_array = state_array[:expected_dim]
            else:
                state_array = np.pad(state_array, (0, expected_dim - state_array.shape[0]))
        
        return state_array
    
    # ============= FIXED SOCKET METHODS FOR WINDOWS =============
    def _check_connections(self):
        """Accept new robot connections - FIXED for Windows non-blocking sockets"""
        if self.server is None:
            return
            
        try:
            ready, _, _ = select.select([self.server], [], [], 0.01)
            if ready:
                try:
                    conn, addr = self.server.accept()
                    conn.setblocking(False)
                    
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
                    conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
                    
                    robot_name = None
                    try:
                        conn.settimeout(1.0)
                        data = conn.recv(4096).decode('utf-8').strip()
                        if data:
                            try:
                                info = json.loads(data)
                                robot_name = info.get('name')
                            except json.JSONDecodeError:
                                pass
                        conn.settimeout(None)
                    except (socket.timeout, BlockingIOError):
                        pass
                    
                    if robot_name is None:
                        robot_name = f"robot{len(self.connections)+1}"
                    
                    self.connections[robot_name] = conn
                    self.robot_states[robot_name] = {}
                    self.connection_attempts[robot_name] = 0
                    
                    try:
                        ack_msg = json.dumps({
                            'status': 'connected',
                            'robot_id': robot_name
                        }) + '\n'
                        conn.sendall(ack_msg.encode())
                        print(f"[Env] ✅ {robot_name} connected from {addr}")
                    except Exception as e:
                        print(f"[Env] Error sending ACK to {robot_name}: {e}")
                        
                except BlockingIOError:
                    pass
                except Exception as e:
                    print(f"[Env] Accept error: {e}")
                    
        except Exception as e:
            if hasattr(e, 'winerror') and e.winerror == 10035:
                pass
            else:
                print(f"[Env] Connection check error: {e}")
    
    def _receive_robot_states(self):
        """Receive states from all connected robots - FIXED for Windows"""
        for robot_id, conn in list(self.connections.items()):
            try:
                ready, _, _ = select.select([conn], [], [], 0.001)
                if not ready:
                    continue
                
                data = b""
                try:
                    conn.settimeout(0.1)
                    chunk = conn.recv(8192)
                    if chunk:
                        data = chunk
                        
                        try:
                            while True:
                                chunk = conn.recv(8192)
                                if not chunk:
                                    break
                                data += chunk
                        except (socket.timeout, BlockingIOError):
                            pass
                    conn.settimeout(None)
                    
                except (socket.timeout, BlockingIOError):
                    conn.settimeout(None)
                    continue
                except ConnectionResetError:
                    self._remove_connection(robot_id)
                    continue
                
                if data:
                    try:
                        data_str = data.decode('utf-8').strip()
                        if data_str:
                            state = json.loads(data_str)
                            self.robot_states[robot_id] = state
                            
                            if 'box_position' in state and robot_id == self.agent_ids[0]:
                                self.box_position = np.array(state['box_position'], dtype=np.float32)
                            if 'box_velocity' in state and robot_id == self.agent_ids[0]:
                                self.box_velocity = np.array(state['box_velocity'], dtype=np.float32)
                            
                            if 'position' in state:
                                pos = np.array(state['position'], dtype=np.float32)
                                self.robot_positions[robot_id] = pos
                                dist_to_box = float(np.linalg.norm(pos - self.box_position))
                                self.robot_dist_to_box[robot_id] = dist_to_box
                                
                                grid_x = int((pos[0] + 20) * 2)
                                grid_y = int((pos[1] + 20) * 2)
                                grid_x = np.clip(grid_x, 0, 99)
                                grid_y = np.clip(grid_y, 0, 99)
                                self.visited_grid[grid_x, grid_y] = 1.0
                                
                    except json.JSONDecodeError:
                        pass
                        
            except Exception as e:
                if robot_id in self.connections:
                    self.connection_attempts[robot_id] = self.connection_attempts.get(robot_id, 0) + 1
                    if self.connection_attempts[robot_id] > 10:
                        self._remove_connection(robot_id)
    
    def _send_actions_and_roles(self, actions, role_weights=None):
        """Send actions AND 3 bio-hybrid signals to robots"""
        for i, robot_id in enumerate(self.agent_ids):
            if robot_id in self.connections:
                try:
                    msg = {
                        'action': actions[i].tolist() if isinstance(actions[i], np.ndarray) else actions[i],
                        'timestamp': time.time()
                    }
                    
                    if role_weights is not None and i < len(role_weights):
                        msg['role_weights'] = role_weights[i].tolist() if isinstance(role_weights[i], np.ndarray) else role_weights[i]
                    
                    # 🐜 Ant Colony signals
                    msg['pheromone'] = float(np.mean(self.pheromone_grid))
                    if robot_id in self.robot_states:
                        pos = self.robot_states[robot_id].get('position', [0, 0])
                        msg['pheromone_gradient'] = self._get_pheromone_gradient(pos)
                    
                    # 🐝 Particle Swarm signals
                    msg['global_best_position'] = self.global_best_position.tolist()
                    if robot_id in self.personal_best_positions:
                        msg['personal_best'] = self.personal_best_positions[robot_id].tolist()
                    
                    # 🦅 Flocking signals
                    if np.any(self.flock_center):
                        msg['flock_center'] = self.flock_center.tolist()
                    if np.any(self.flock_velocity):
                        msg['flock_velocity'] = self.flock_velocity.tolist()
                    
                    # Neighbor and visited info
                    msg['neighbors'] = self._get_neighbor_info(robot_id)
                    if robot_id in self.robot_states:
                        pos = self.robot_states[robot_id].get('position', [0, 0])
                        msg['visited_intensity'] = self._get_visited_intensity(pos)
                    
                    try:
                        self.connections[robot_id].sendall(json.dumps(msg).encode() + b'\n')
                    except (ConnectionResetError, BrokenPipeError):
                        self._remove_connection(robot_id)
                    
                except Exception as e:
                    self._remove_connection(robot_id)
    
    # ============= HELPER METHODS =============
    def _get_neighbor_info(self, robot_id):
        """Get neighbor information for cooperation"""
        neighbors = []
        if robot_id not in self.robot_positions:
            return neighbors
        
        robot_pos = self.robot_positions[robot_id]
        
        for other_id, other_pos in self.robot_positions.items():
            if other_id != robot_id:
                dist = float(np.linalg.norm(robot_pos - other_pos))
                if dist < 5.0:
                    dist_to_box = self.robot_dist_to_box.get(other_id, 10.0)
                    neighbors.append({
                        'id': other_id,
                        'x': float(other_pos[0]),
                        'y': float(other_pos[1]),
                        'distance': float(dist),
                        'dist_to_box': float(dist_to_box),
                        'needs_help': dist_to_box > 3.0
                    })
        
        return neighbors
    
    def _get_visited_intensity(self, position):
        """Get visited intensity for exploration"""
        x, y = position
        grid_x = int((x + 20) * 2)
        grid_y = int((y + 20) * 2)
        grid_x = np.clip(grid_x, 0, 99)
        grid_y = np.clip(grid_y, 0, 99)
        return float(self.visited_grid[grid_x, grid_y])
    
    def _remove_connection(self, robot_id):
        """Cleanly remove a dead connection"""
        try:
            if robot_id in self.connections:
                try:
                    self.connections[robot_id].close()
                except:
                    pass
                del self.connections[robot_id]
            
            if robot_id in self.robot_states:
                del self.robot_states[robot_id]
            if robot_id in self.robot_positions:
                del self.robot_positions[robot_id]
            if robot_id in self.robot_dist_to_box:
                del self.robot_dist_to_box[robot_id]
            if robot_id in self.personal_best_positions:
                del self.personal_best_positions[robot_id]
            if robot_id in self.personal_best_rewards:
                del self.personal_best_rewards[robot_id]
            if robot_id in self.pso_velocity:
                del self.pso_velocity[robot_id]
            if robot_id in self.connection_attempts:
                del self.connection_attempts[robot_id]
                
            print(f"[Env] 🔌 {robot_id} disconnected")
        except:
            pass
    
    # ============= ENVIRONMENT CORE METHODS =============
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        self.step_count = 0
        self.episode_reward = 0.0
        self.episode_collisions = 0
        self.episode_cooperations = 0
        self.episode_count += 1
        
        print(f"\n[Env] Episode {self.episode_count} - Waiting for robots to connect...")
        
        self.robot_states = {}
        self.robot_positions = {}
        self.robot_dist_to_box = {}
        self.personal_best_positions = {}
        self.personal_best_rewards = {}
        self.pso_velocity = {}
        self.connection_attempts = {}
        
        start_time = time.time()
        timeout = 60
        
        while len(self.connections) < self.num_agents:
            self._check_connections()
            if time.time() - start_time > timeout:
                print(f"[Env] ⚠️ Timeout! Only {len(self.connections)}/{self.num_agents} robots connected")
                break
            time.sleep(0.1)
        
        print(f"[Env] {len(self.connections)}/{self.num_agents} robots ready")
        
        self.pheromone_grid = np.zeros((100, 100))
        self.visited_grid = np.zeros((100, 100))
        
        self.global_best_position = np.array([12.0, 0.0])
        self.global_best_reward = -float('inf')
        self.flock_center = np.array([0.0, 0.0])
        self.flock_velocity = np.array([0.0, 0.0])
        
        for robot_id, conn in self.connections.items():
            try:
                reset_msg = json.dumps({
                    'type': 'reset',
                    'episode': self.episode_count
                }) + '\n'
                conn.sendall(reset_msg.encode())
            except:
                self._remove_connection(robot_id)
        
        time.sleep(0.5)
        self._receive_robot_states()
        obs = self._get_observations()
        
        try:
            state = self._get_centralized_state()
        except Exception as e:
            print(f"⚠️ Error getting state: {e}, using zeros")
            state_dim = self.num_agents * 2 + 4 + 2 + 20
            state = np.zeros(state_dim, dtype=np.float32)
        
        return obs, {"state": state}
    
    def step(self, actions, role_weights=None):
        """Step environment - FIXED shape handling"""
        self.step_count += 1
        
        if isinstance(actions, np.ndarray):
            if actions.shape != (self.num_agents, 2):
                try:
                    actions = actions.reshape(self.num_agents, 2)
                except:
                    actions = np.array(actions).reshape(self.num_agents, 2)
        
        if role_weights is not None:
            if isinstance(role_weights, np.ndarray):
                if len(role_weights.shape) == 1:
                    expected_len = self.num_agents * self.num_roles
                    if role_weights.shape[0] == expected_len:
                        role_weights = role_weights.reshape(self.num_agents, self.num_roles)
                    else:
                        role_weights = role_weights.reshape(self.num_agents, -1)
                elif role_weights.shape != (self.num_agents, self.num_roles):
                    role_weights = role_weights.reshape(self.num_agents, self.num_roles)
        
        self._send_actions_and_roles(actions, role_weights)
        
        self._check_connections()
        self._receive_robot_states()
        
        reward = self._calculate_reward(role_weights)
        self.episode_reward += reward
        
        obs = self._get_observations()
        state = self._get_centralized_state()
        
        terminated = False
        truncated = False
        
        box_np = np.array([float(self.box_position[0]), float(self.box_position[1])])
        goal_np = np.array([float(self.goal_x), float(self.goal_y)])
        distance_to_goal = float(np.linalg.norm(box_np - goal_np))
        
        if distance_to_goal < self.goal_radius:
            terminated = True
            reward += 100.0
            print(f"\n[Env] ✓ SUCCESS! Box reached goal!")
        
        elif self.step_count >= self.max_steps:
            truncated = True
        
        touching_box = 0
        for robot_id in self.agent_ids:
            if robot_id in self.robot_states:
                pos = self.robot_states[robot_id].get('position', [0, 0])
                pos_np = np.array([float(pos[0]), float(pos[1])])
                dist = float(np.linalg.norm(pos_np - box_np))
                if dist < 2.0:
                    touching_box += 1
        
        info = {
            'step': self.step_count,
            'episode': self.episode_count,
            'box_x': float(self.box_position[0]),
            'box_y': float(self.box_position[1]),
            'box_velocity': float(self.box_velocity[0]),
            'progress': float(self.box_start_x - self.box_position[0]),
            'active_robots': len(self.connections),
            'touching_box': touching_box,
            'cooperations': self.episode_cooperations,
            'collisions': self.episode_collisions,
            'reward': float(reward),
            'episode_reward': float(self.episode_reward),
            'state': state,
            'observations': obs
        }
        
        if self.step_count % 20 == 0:
            log_msg = (f"[Step {self.step_count:3d}] Box: ({self.box_position[0]:5.2f}) | "
                      f"Progress: {info['progress']:5.2f} | "
                      f"Active: {len(self.connections)}/{self.num_agents} | "
                      f"Reward: {reward:6.2f}")
            print(log_msg)
        
        return obs, reward, terminated, truncated, info
    
    def close(self):
        """Clean shutdown"""
        print("[Env] Shutting down...")
        
        for robot_id, conn in self.connections.items():
            try:
                conn.sendall(json.dumps({'type': 'shutdown'}).encode() + b'\n')
                time.sleep(0.1)
                conn.close()
            except:
                pass
        
        if self.server:
            try:
                self.server.close()
            except:
                pass
        
        print("[Env] Clean shutdown complete")