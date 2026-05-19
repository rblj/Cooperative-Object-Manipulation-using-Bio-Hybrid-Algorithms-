"""
Bio-Hybrid MAPPO Algorithm for Cooperative Box Pushing
Centralized Training with Decentralized Execution (CTDE)
No supervisor during execution - pure RL with bio-hybrid reward shaping
Outputs BOTH actions AND role weights for 3 bio-hybrid algorithms!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import deque
import os
import json
import time

# ============================================
# DEVICE CONFIGURATION
# ============================================
def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()
print(f"🐍 Bio-Hybrid MAPPO using device: {DEVICE}")


# ============================================
# CENTRALIZED CRITIC
# ============================================
class CentralizedCritic(nn.Module):
    """
    Centralized Critic Network with Multi-Head Attention
    Sees global state of ALL agents and environment
    Used ONLY during training - NEVER during execution!
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 256, num_heads: int = 4):
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        print(f"[CentralizedCritic] Initializing with state_dim: {state_dim}")
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.to(DEVICE)
    
    def forward(self, state: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(np.array(state)).to(DEVICE)
        else:
            state = state.to(DEVICE)
        
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        assert state.shape[1] == self.state_dim, f"State dim mismatch: got {state.shape[1]}, expected {self.state_dim}"
        
        features = self.state_encoder(state)
        value = self.value_net(features)
        return value


# ============================================
# DECENTRALIZED ACTOR with 3 ROLE WEIGHTS
# ============================================
class DecentralizedActor(nn.Module):
    """
    Decentralized Actor Network
    Outputs BOTH actions AND role weights for 3 bio-hybrid algorithms!
    role_weights: [ant_colony, particle_swarm, flocking]
    """
    
    def __init__(self, 
                 obs_dim: int, 
                 action_dim: int = 2, 
                 num_roles: int = 3,  # CHANGED FROM 5 TO 3
                 hidden_dim: int = 128,
                 log_std_init: float = -0.5):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_roles = num_roles
        self.hidden_dim = hidden_dim
        
        print(f"[DecentralizedActor] Initializing with obs_dim: {obs_dim}, num_roles: {num_roles}")
        print(f"[DecentralizedActor] Roles: 🐜 Ant | 🐝 PSO | 🦅 Flock")
        
        self.features = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        # ROLE WEIGHT HEAD - 3 weights for 3 bio-hybrid algorithms!
        self.role_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_roles),
            nn.Sigmoid()
        )
        
        self.log_std = nn.Parameter(
            torch.ones(action_dim) * log_std_init,
            requires_grad=True
        )
        
        self.to(DEVICE)
    
    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(np.array(obs)).to(DEVICE)
        else:
            obs = obs.to(DEVICE)
        
        original_shape = obs.shape
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        
        if obs.shape[1] != self.obs_dim:
            if len(original_shape) == 1:
                total_elements = obs.shape[0] * obs.shape[1] if len(obs.shape) > 1 else obs.shape[0]
                expected_elements = self.obs_dim
                if total_elements % expected_elements == 0:
                    batch_size = total_elements // expected_elements
                    obs = obs.reshape(batch_size, self.obs_dim)
                else:
                    if obs.shape[1] > self.obs_dim:
                        obs = obs[:, :self.obs_dim]
                    else:
                        padding = torch.zeros(obs.shape[0], self.obs_dim - obs.shape[1]).to(DEVICE)
                        obs = torch.cat([obs, padding], dim=1)
        
        features = self.features(obs)
        
        action_mean = self.action_head(features)
        role_weights = self.role_head(features)
        
        log_std = self.log_std.expand_as(action_mean)
        std = torch.exp(torch.clamp(log_std, -20, 2))
        
        return action_mean, std, role_weights
    
    def get_distribution_and_roles(self, obs: Union[np.ndarray, torch.Tensor]) -> Tuple[torch.distributions.Normal, torch.Tensor]:
        mean, std, role_weights = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        return dist, role_weights


# ============================================
# EXPERIENCE BUFFER with 3 ROLE WEIGHTS
# ============================================
class RolloutBuffer:
    """
    Rollout buffer for MAPPO with centralized critic
    Stores experiences for PPO update - WITH 3 ROLE WEIGHTS!
    """
    
    def __init__(self, 
                 buffer_size: int = 2000,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95):
        
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.reset()
    
    def reset(self):
        """Reset buffer"""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.states = []
        self.role_weights = []
        self.advantages = []
        self.returns = []
        
        self.pos = 0
    
    def add(self, 
           obs: np.ndarray,
           actions: np.ndarray,
           reward: float,
           done: bool,
           value: float,
           log_prob: np.ndarray,
           state: np.ndarray,
           role_weights: Optional[np.ndarray] = None):
        """Add transition to buffer with 3 role weights"""
        self.observations.append(obs)
        self.actions.append(actions)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.states.append(state)
        self.role_weights.append(role_weights)
        
        self.pos += 1
        
        if self.pos > self.buffer_size:
            self.observations.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.dones.pop(0)
            self.values.pop(0)
            self.log_probs.pop(0)
            self.states.pop(0)
            self.role_weights.pop(0)
            self.pos = self.buffer_size
    
    def compute_advantages(self, last_value: float, last_done: bool):
        """Compute GAE advantages - advantages are per-episode"""
        advantages = []
        gae = 0
        values = self.values + [last_value]
        
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + \
                    self.gamma * values[t + 1] * (1 - self.dones[t]) - \
                    values[t]
            
            gae = delta + \
                self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            
            advantages.insert(0, gae)
        
        self.advantages = advantages
        self.returns = [adv + val for adv, val in zip(advantages, self.values)]
        
        return advantages
    
    def get_training_batch(self, batch_size: int = 64) -> Dict:
        """Get a random batch for training"""
        if len(self.observations) < batch_size:
            return None
        
        indices = np.random.choice(len(self.observations), batch_size, replace=False)
        
        observations = []
        actions = []
        log_probs = []
        states = []
        rewards = []
        dones = []
        values = []
        role_weights = []
        
        for i in indices:
            observations.append(self.observations[i])
            actions.append(self.actions[i])
            log_probs.append(self.log_probs[i])
            states.append(self.states[i])
            rewards.append(self.rewards[i])
            dones.append(self.dones[i])
            values.append(self.values[i])
            if self.role_weights and self.role_weights[i] is not None:
                role_weights.append(self.role_weights[i])
        
        batch = {
            'observations': np.array(observations),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'dones': np.array(dones),
            'values': np.array(values),
            'log_probs': np.array(log_probs),
            'states': np.array(states),
            'role_weights': np.array(role_weights) if role_weights else None,
            'advantages': np.array([self.advantages[i] for i in indices]),
            'returns': np.array([self.returns[i] for i in indices])
        }
        
        return batch
    
    def __len__(self):
        return len(self.observations)


# ============================================
# MAIN MAPPO ALGORITHM with 3 ROLE WEIGHTS
# ============================================
class BioHybridMAPPO:
    """
    Multi-Agent PPO with Bio-Hybrid Cooperation
    Centralized Training with Decentralized Execution (CTDE)
    
    Features 3 bio-hybrid algorithms:
    - 🐜 Ant Colony Optimization
    - 🐝 Particle Swarm Optimization
    - 🦅 Flocking (Boids)
    """
    
    def __init__(self,
                 env,
                 bio_algorithm: str = 'ant_colony',
                 num_agents: int = 5,
                 obs_dim: Optional[int] = None,
                 state_dim: Optional[int] = None,
                 action_dim: int = 2,
                 num_roles: int = 3,  # CHANGED FROM 5 TO 3
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 n_epochs: int = 10,
                 batch_size: int = 64,
                 buffer_size: int = 1000,
                 model_dir: str = "mappo_models"):
        
        self.env = env
        self.bio_algorithm = bio_algorithm
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.num_roles = num_roles
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.model_dir = model_dir
        
        self.device = DEVICE
        
        if obs_dim is None:
            if hasattr(env, 'observation_space'):
                if hasattr(env.observation_space, 'shape'):
                    obs_dim = env.observation_space.shape[0]
                    print(f"[BioHybridMAPPO] Got obs_dim from env: {obs_dim}")
                else:
                    obs_dim = 23  # UPDATED: 4+4+2+2+8+3 = 23 (3 bio signals)
                    print(f"[BioHybridMAPPO] Using default obs_dim: {obs_dim}")
            else:
                obs_dim = 23
                print(f"[BioHybridMAPPO] Using default obs_dim: {obs_dim}")
        
        if state_dim is None:
            if hasattr(env, 'state_space'):
                if hasattr(env.state_space, 'shape'):
                    state_dim = env.state_space.shape[0]
                    print(f"[BioHybridMAPPO] Got state_dim from env: {state_dim}")
                else:
                    # UPDATED for 3 algorithms: num_agents*2 + 4 + 2 + 20
                    state_dim = num_agents * 2 + 4 + 2 + 20
                    print(f"[BioHybridMAPPO] Using calculated state_dim: {state_dim}")
            else:
                state_dim = num_agents * 2 + 4 + 2 + 20
                print(f"[BioHybridMAPPO] Using calculated state_dim: {state_dim}")
        
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        
        print(f"\n[BioHybridMAPPO] 🧬 Initializing with 3 bio-hybrid algorithms")
        print(f"[BioHybridMAPPO] Current focus: {bio_algorithm.upper()}")
        print(f"[BioHybridMAPPO] Observation dim: {obs_dim}, State dim: {state_dim}")
        print(f"[BioHybridMAPPO] Number of roles: {num_roles} (3 algorithms)")
        print(f"[BioHybridMAPPO] Device: {self.device}")
        print(f"[BioHybridMAPPO] 🐜 Ant | 🐝 PSO | 🦅 Flock")
        
        self.actor = DecentralizedActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_roles=num_roles,
            hidden_dim=128
        ).to(self.device)
        
        self.critic = CentralizedCritic(
            state_dim=state_dim,
            hidden_dim=256,
            num_heads=4
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr
        )
        
        self.buffer = RolloutBuffer(
            buffer_size=buffer_size,
            gamma=gamma,
            gae_lambda=gae_lambda
        )
        
        os.makedirs(model_dir, exist_ok=True)
        
        self.training_step = 0
        self.episode_count = 0
        
        self.execution_mode = False
    
    def set_execution_mode(self, mode: bool = True):
        """Switch to execution mode - NO SUPERVISOR!"""
        self.execution_mode = mode
        if mode:
            print("\n" + "=" * 60)
            print("🤖 EXECUTION MODE ACTIVATED")
            print("✅ NO supervisor - each agent acts independently")
            print("✅ NO centralized critic - using only local observations")
            print("✅ NO server required - fully decentralized")
            print("=" * 60 + "\n")
    
    def select_actions(self, 
                      observations: Union[np.ndarray, torch.Tensor], 
                      deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Select actions AND 3 role weights for all agents
        Returns (num_agents, action_dim) and (num_agents, num_roles)
        """
        if not isinstance(observations, torch.Tensor):
            observations = torch.FloatTensor(np.array(observations)).to(self.device)
        else:
            observations = observations.to(self.device)
        
        if len(observations.shape) == 1:
            try:
                observations = observations.reshape(self.num_agents, self.obs_dim)
            except:
                observations = observations.reshape(1, -1)
        elif len(observations.shape) == 2:
            if observations.shape[0] == self.num_agents:
                if observations.shape[1] != self.obs_dim:
                    if observations.shape[1] > self.obs_dim:
                        observations = observations[:, :self.obs_dim]
                    else:
                        padding = torch.zeros(observations.shape[0], self.obs_dim - observations.shape[1]).to(self.device)
                        observations = torch.cat([observations, padding], dim=1)
            else:
                try:
                    observations = observations.reshape(self.num_agents, self.obs_dim)
                except:
                    pass
        elif len(observations.shape) == 3:
            observations = observations[0]
        
        with torch.no_grad():
            dist, role_weights = self.actor.get_distribution_and_roles(observations)
            
            if deterministic:
                actions = dist.mean
            else:
                actions = dist.sample()
            
            log_probs = dist.log_prob(actions).sum(dim=-1)
        
        actions_np = actions.cpu().numpy()
        log_probs_np = log_probs.cpu().numpy()
        role_weights_np = role_weights.cpu().numpy()
        
        if len(actions_np.shape) == 1:
            actions_np = actions_np.reshape(1, -1)
        if len(role_weights_np.shape) == 1:
            role_weights_np = role_weights_np.reshape(1, -1)
        if len(log_probs_np.shape) == 0:
            log_probs_np = np.array([log_probs_np])
        
        return actions_np, log_probs_np, role_weights_np
    
    def get_value(self, state: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Get state value estimate from centralized critic - TRAINING ONLY!"""
        if self.execution_mode:
            raise RuntimeError("❌ CRITICAL ERROR: Trying to use centralized critic during execution!")
        return self.critic(state)
    
    def store_transition(self, 
                        obs: np.ndarray,
                        actions: np.ndarray,
                        reward: float,
                        done: bool,
                        value: float,
                        log_prob: np.ndarray,
                        state: np.ndarray,
                        role_weights: Optional[np.ndarray] = None):
        """Store transition in experience buffer WITH 3 role weights"""
        self.buffer.add(obs, actions, reward, done, value, log_prob, state, role_weights)
    
    def update(self, next_state: np.ndarray, last_done: bool) -> Dict[str, float]:
        """Update policy and value networks using PPO"""
        
        if len(self.buffer) < self.batch_size:
            return {'actor_loss': 0, 'critic_loss': 0, 'entropy': 0, 'clip_fraction': 0}
        
        if isinstance(next_state, np.ndarray):
            if len(next_state.shape) == 1:
                if next_state.shape[0] != self.state_dim:
                    if next_state.shape[0] > self.state_dim:
                        next_state = next_state[:self.state_dim]
                    else:
                        next_state = np.pad(next_state, (0, self.state_dim - next_state.shape[0]))
        
        with torch.no_grad():
            try:
                last_value = self.get_value(next_state).squeeze().cpu().numpy()
                if isinstance(last_value, np.ndarray):
                    last_value = last_value.item() if last_value.size == 1 else last_value[0]
            except Exception as e:
                print(f"⚠️ Error getting last value: {e}, using 0")
                last_value = 0.0
        
        self.buffer.compute_advantages(last_value, last_done)
        
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        total_clip_frac = 0
        n_batches = 0
        
        for _ in range(self.n_epochs):
            batch = self.buffer.get_training_batch(self.batch_size)
            if batch is None:
                break
            
            observations = batch['observations']
            batch_size = observations.shape[0]
            
            try:
                observations = observations.reshape(batch_size * self.num_agents, self.obs_dim)
            except Exception as e:
                print(f"⚠️ Failed to reshape observations: {e}")
                continue
            
            actions = batch['actions']
            try:
                actions = actions.reshape(batch_size * self.num_agents, self.action_dim)
            except Exception as e:
                print(f"⚠️ Failed to reshape actions: {e}")
                continue
            
            old_log_probs = batch['log_probs']
            try:
                old_log_probs = old_log_probs.reshape(batch_size * self.num_agents)
            except Exception as e:
                old_log_probs = old_log_probs.flatten()
            
            states = batch['states']
            advantages = batch['advantages']
            returns = batch['returns']
            
            observations = torch.FloatTensor(observations).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
            states = torch.FloatTensor(states).to(self.device)
            advantages = torch.FloatTensor(advantages).to(self.device)
            returns = torch.FloatTensor(returns).to(self.device)
            
            expanded_advantages = advantages.unsqueeze(1).expand(-1, self.num_agents).reshape(-1)
            
            if expanded_advantages.std() > 0:
                expanded_advantages = (expanded_advantages - expanded_advantages.mean()) / (expanded_advantages.std() + 1e-8)
            
            dist, _ = self.actor.get_distribution_and_roles(observations)
            
            action_log_probs = dist.log_prob(actions).sum(dim=-1)
            
            assert action_log_probs.shape == old_log_probs.shape, \
                f"Shape mismatch: action_log_probs {action_log_probs.shape} vs old_log_probs {old_log_probs.shape}"
            assert action_log_probs.shape == expanded_advantages.shape, \
                f"Shape mismatch: action_log_probs {action_log_probs.shape} vs advantages {expanded_advantages.shape}"
            
            ratio = torch.exp(action_log_probs - old_log_probs)
            
            surr1 = ratio * expanded_advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * expanded_advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            values = self.critic(states).squeeze()
            critic_loss = F.mse_loss(values, returns)
            
            entropy = dist.entropy().sum(dim=-1).mean()
            entropy_loss = -entropy
            
            loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                self.max_grad_norm
            )
            
            self.optimizer.step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.item()
            total_clip_frac += ((ratio < 1 - self.clip_epsilon).float().mean().item() + 
                            (ratio > 1 + self.clip_epsilon).float().mean().item()) / 2
            n_batches += 1
            self.training_step += 1
        
        if n_batches > 0:
            self.buffer.reset()
        
        return {
            'actor_loss': total_actor_loss / max(n_batches, 1),
            'critic_loss': total_critic_loss / max(n_batches, 1),
            'entropy': total_entropy / max(n_batches, 1),
            'clip_fraction': total_clip_frac / max(n_batches, 1)
        }
    
    def save_model(self, path: Optional[str] = None):
        """Save model checkpoint"""
        if path is None:
            path = f"{self.model_dir}/mappo_bio_{self.bio_algorithm}.pt"
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'bio_algorithm': self.bio_algorithm,
            'obs_dim': self.obs_dim,
            'state_dim': self.state_dim,
            'num_agents': self.num_agents,
            'num_roles': self.num_roles
        }, path)
        
        print(f"[MAPPO] Model saved to {path}")
    
    def load_model(self, path: str) -> bool:
        """Load model checkpoint"""
        if not os.path.exists(path):
            print(f"[MAPPO] No model found at {path}")
            return False
        
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_step = checkpoint.get('training_step', 0)
            self.bio_algorithm = checkpoint.get('bio_algorithm', self.bio_algorithm)
            
            print(f"[MAPPO] Model loaded from {path}")
            return True
            
        except Exception as e:
            print(f"[MAPPO] Error loading model: {e}")
            return False


# ============================================
# INFERENCE WRAPPER for DEPLOYMENT
# ============================================
class MAPPOInference:
    """
    Inference wrapper for deployed MAPPO model
    NO supervisor, NO server, fully decentralized!
    Outputs actions AND 3 role weights!
    """
    
    def __init__(self, model_path: str, num_agents: int = 5, obs_dim: int = 23, num_roles: int = 3):
        """
        Load trained model for deployment
        
        Args:
            model_path: Path to saved model
            num_agents: Number of agents
            obs_dim: Observation dimension (23 for 3 bio signals + lidar)
            num_roles: Number of role weights (3 for 3 algorithms)
        """
        self.device = DEVICE
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = 2
        self.num_roles = num_roles
        
        print(f"[MAPPOInference] Loading model with obs_dim={obs_dim}, num_roles={num_roles}")
        print(f"[MAPPOInference] Roles: 🐜 Ant | 🐝 PSO | 🦅 Flock")
        
        self.actor = DecentralizedActor(
            obs_dim=obs_dim,
            action_dim=self.action_dim,
            num_roles=num_roles,
            hidden_dim=128
        ).to(self.device)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'actor_state_dict' in checkpoint:
                self.actor.load_state_dict(checkpoint['actor_state_dict'])
                print(f"✅ Model loaded from {model_path}")
            else:
                print(f"⚠️ No actor state dict found in {model_path}")
        else:
            print(f"❌ Model not found at {model_path}")
        
        self.actor.eval()
        
        print("🤖 MAPPO Inference Ready - NO SUPERVISOR MODE")
        print("✅ Each agent acts independently using only local observations")
        print("✅ Outputs actions AND 3 role weights:")
        print("   🐜 Ant | 🐝 PSO | 🦅 Flock")
    
    def predict(self, 
               observation: Union[np.ndarray, torch.Tensor], 
               deterministic: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict action AND 3 role weights from local observation
        """
        if not isinstance(observation, torch.Tensor):
            observation = torch.FloatTensor(np.array(observation)).to(self.device)
        else:
            observation = observation.to(self.device)
        
        if len(observation.shape) == 1:
            observation = observation.unsqueeze(0)
        
        with torch.no_grad():
            dist, role_weights = self.actor.get_distribution_and_roles(observation)
            
            if deterministic:
                actions = dist.mean
            else:
                actions = dist.sample()
        
        if len(actions) == 1:
            return actions.squeeze().cpu().numpy(), role_weights.squeeze().cpu().numpy()
        else:
            return actions.cpu().numpy(), role_weights.cpu().numpy()


# ============================================
# EXPORTS
# ============================================
__all__ = [
    'DEVICE',
    'CentralizedCritic',
    'DecentralizedActor',
    'RolloutBuffer',
    'BioHybridMAPPO',
    'MAPPOInference'
]