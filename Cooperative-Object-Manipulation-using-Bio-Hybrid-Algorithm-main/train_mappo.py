import os
import numpy as np
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from swarm_env import SwarmEnvironment
from mappo_algorithm import BioHybridMAPPO, DEVICE
from collections import deque
import argparse
import traceback
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Train MAPPO for box pushing with 3 bio-hybrid algorithms')
    parser.add_argument('--bio', type=str, default='ant_colony',
                       choices=['ant_colony', 'particle_swarm', 'flocking', 'all'],
                       help='Bio-hybrid algorithm to focus on')
    parser.add_argument('--agents', type=int, default=5, help='Number of agents')
    parser.add_argument('--steps', type=int, default=100000, help='Total training steps')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--load', type=str, default=None, help='Load model from path')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--roles', type=int, default=3, help='Number of role weights (3 for 3 algorithms)')
    parser.add_argument('--port', type=int, default=12345, help='Socket port')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--buffer-size', type=int, default=1000, help='Buffer size')
    parser.add_argument('--log-dir', type=str, default='./logs', help='TensorBoard log directory')
    parser.add_argument('--run-name', type=str, default=None, help='Name for this run (default: timestamp)')
    return parser.parse_args()

def get_value_safe(model, state):
    """
    Safely get value from model with proper device handling and NO gradients
    Returns a scalar value (not array)
    """
    try:
        if hasattr(model, 'get_value'):
            value = model.get_value(state)
            value = value.detach().cpu().numpy().squeeze()
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    value = value.item()
                else:
                    value = value[0]
            return float(value)
        
        elif hasattr(model, 'critic'):
            if not isinstance(state, torch.Tensor):
                state_tensor = torch.FloatTensor(np.array(state)).to(model.device)
            else:
                state_tensor = state.to(model.device)
            
            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            with torch.no_grad():
                value = model.critic(state_tensor)
            
            value = value.detach().cpu().numpy().squeeze()
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    value = value.item()
                else:
                    value = value[0]
            return float(value)
        
        else:
            return 0.0
            
    except Exception as e:
        return 0.0

def main():
    args = parse_args()
    
    # Set device
    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ============= TENSORBOARD SETUP =============
    # Create unique run name
    if args.run_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{args.bio}_agents{args.agents}_{timestamp}"
    else:
        run_name = args.run_name
    
    # Create log directory
    log_dir = os.path.join(args.log_dir, run_name)
    writer = SummaryWriter(log_dir)
    
    print("=" * 70)
    print("🧬 MAPPO TRAINING WITH 3 BIO-HYBRID ALGORITHMS")
    print("=" * 70)
    print(f"Algorithms: 🐜 Ant | 🐝 PSO | 🦅 Flock")
    print(f"Focus: {args.bio.upper()}")
    print(f"Agents: {args.agents} | Steps: {args.steps} | LR: {args.lr}")
    print(f"Roles: {args.roles} (3 algorithms) | Device: {device}")
    print(f"Port: {args.port} | Batch Size: {args.batch_size} | Buffer Size: {args.buffer_size}")
    print(f"TensorBoard: {log_dir}")
    print("=" * 70)
    
    # Create directories
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Save hyperparameters to TensorBoard
    hparams = {
        'bio_algorithm': args.bio,
        'num_agents': args.agents,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'buffer_size': args.buffer_size,
        'num_roles': args.roles,
        'max_steps': args.steps,
        'device': str(device)
    }
    writer.add_hparams(hparams, {'hparam/placeholder': 0})
    
    # Create environment
    print("\n>>> Creating environment...")
    env = SwarmEnvironment(
        num_agents=args.agents,
        port=args.port,
        use_lidar=True,
        bio_algorithm=args.bio,
        max_steps=300
    )
    
    # Reset to get dimensions
    print(">>> Waiting for robots to connect...")
    obs, info = env.reset()
    
    # Get observation dimension
    if isinstance(obs, np.ndarray):
        if len(obs.shape) == 2:
            obs_dim = obs.shape[1]
            print(f"✅ Observations shape: {obs.shape} = {obs_dim} dims per agent")
        elif len(obs.shape) == 1:
            obs_dim = obs.shape[0] // args.agents
            print(f"✅ Observations flattened: {obs.shape}, reshaping to {args.agents} x {obs_dim}")
        else:
            obs_dim = obs.shape[0]
            print(f"✅ Observations shape: {obs.shape}")
    else:
        obs_dim = 23
        print(f"⚠️ Using default observation dim: {obs_dim}")
    
    # Get state dimension
    if isinstance(info, dict) and 'state' in info:
        if isinstance(info['state'], np.ndarray):
            state_dim = info['state'].shape[0]
            print(f"✅ State dimension: {state_dim}")
        else:
            state_dim = args.agents * 2 + 4 + 2 + 20
            print(f"⚠️ State not array, using calculated dim: {state_dim}")
    else:
        state_dim = args.agents * 2 + 4 + 2 + 20
        print(f"⚠️ Using calculated state dim: {state_dim}")
    
    # Create model - WITH 3 ROLE WEIGHTS!
    model = BioHybridMAPPO(
        env=env,
        bio_algorithm=args.bio,
        num_agents=args.agents,
        obs_dim=obs_dim,
        state_dim=state_dim,
        num_roles=args.roles,
        lr=args.lr,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        n_epochs=10,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        model_dir=f"./models/mappo_3algo_{args.bio}"
    )
    
    # Load model if specified
    if args.load:
        load_path = args.load
        if os.path.exists(load_path):
            if model.load_model(load_path):
                print(f">>> Loaded model from {load_path}")
        else:
            print(f">>> No model found at {load_path}, starting fresh")
    
    print(f"\n>>> Starting training with 3 bio-hybrid algorithms...")
    print(f">>> Each agent learns actions AND 3 role weights:")
    print(f"    [🐜 Ant, 🐝 PSO, 🦅 Flock]")
    print(f">>> TensorBoard: run 'tensorboard --logdir {args.log_dir}'")
    print(">>> Press Ctrl+C to stop\n")
    
    # Training loop
    episode = 0
    episode_reward = 0
    episode_steps = 0
    total_successes = 0
    success_count = 0
    
    # Metrics tracking
    episode_rewards = deque(maxlen=20)
    cooperation_rates = deque(maxlen=20)
    collision_rates = deque(maxlen=20)
    role_weight_history = deque(maxlen=100)
    box_position_history = deque(maxlen=100)
    progress_history = deque(maxlen=100)
    active_robots_history = deque(maxlen=100)
    
    # Role weight names for logging
    role_names = ['Ant', 'PSO', 'Flock']
    role_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # Training start time
    train_start_time = time.time()
    
    # Reset environment
    obs, info = env.reset()
    
    try:
        for step in range(args.steps):
            # Select actions AND 3 role weights
            actions, log_probs, role_weights = model.select_actions(obs)
            
            if step % 100 == 0:
                print(f"\n[DEBUG] Step {step}:")
                print(f"  obs shape: {obs.shape if hasattr(obs, 'shape') else 'unknown'}")
                print(f"  actions shape: {actions.shape}")
                print(f"  role_weights shape: {role_weights.shape if role_weights is not None else 'None'}")
            
            if role_weights is not None:
                if len(role_weights.shape) == 1:
                    expected_len = args.agents * args.roles
                    if role_weights.shape[0] == expected_len:
                        role_weights = role_weights.reshape(args.agents, args.roles)
                    else:
                        role_weights = role_weights.reshape(args.agents, -1)
                elif len(role_weights.shape) == 2:
                    if role_weights.shape[0] != args.agents or role_weights.shape[1] != args.roles:
                        role_weights = role_weights.reshape(args.agents, args.roles)
            
            if len(actions.shape) == 1:
                expected_len = args.agents * 2
                if actions.shape[0] == expected_len:
                    actions = actions.reshape(args.agents, 2)
                else:
                    actions = actions.reshape(args.agents, -1)
            
            assert actions.shape == (args.agents, 2), f"Actions shape {actions.shape} != ({args.agents}, 2)"
            if role_weights is not None:
                assert role_weights.shape == (args.agents, args.roles), f"Role weights shape {role_weights.shape} != ({args.agents}, {args.roles})"
            
            next_obs, reward, terminated, truncated, info = env.step(actions, role_weights)
            done = terminated or truncated
            
            if 'state' in info and info['state'] is not None:
                value = get_value_safe(model, info['state'])
            else:
                value = 0.0
            
            if isinstance(reward, np.ndarray):
                reward_sum = np.sum(reward)
            else:
                reward_sum = float(reward)
            
            if isinstance(log_probs, np.ndarray):
                if len(log_probs.shape) == 1:
                    log_probs_flat = log_probs
                else:
                    log_probs_flat = log_probs.flatten()
            else:
                log_probs_flat = np.zeros(args.agents)
            
            if isinstance(obs, np.ndarray):
                if len(obs.shape) == 2:
                    obs_flat = obs.flatten()
                else:
                    obs_flat = obs.flatten()
            else:
                obs_flat = np.zeros(obs_dim * args.agents)
            
            if isinstance(actions, np.ndarray):
                if len(actions.shape) == 2:
                    actions_flat = actions.flatten()
                else:
                    actions_flat = actions.flatten()
            else:
                actions_flat = np.zeros(args.agents * 2)
            
            if 'state' in info and info['state'] is not None:
                state = info['state']
                if isinstance(state, np.ndarray):
                    if state.shape[0] != state_dim:
                        if state.shape[0] > state_dim:
                            state = state[:state_dim]
                        else:
                            state = np.pad(state, (0, state_dim - state.shape[0]))
                else:
                    state = np.zeros(state_dim)
            else:
                state = np.zeros(state_dim)
            
            if role_weights is not None:
                if isinstance(role_weights, np.ndarray):
                    role_flat = role_weights.flatten()
                    expected_len = args.agents * args.roles
                    if len(role_flat) != expected_len:
                        if len(role_flat) > expected_len:
                            role_flat = role_flat[:expected_len]
                        else:
                            role_flat = np.pad(role_flat, (0, expected_len - len(role_flat)))
                else:
                    role_flat = np.zeros(args.agents * args.roles)
            else:
                role_flat = np.zeros(args.agents * args.roles)
            
            model.store_transition(
                obs_flat,
                actions_flat,
                reward_sum,
                done,
                value,
                log_probs_flat,
                state,
                role_flat
            )
            
            episode_reward += reward_sum
            obs = next_obs
            episode_steps += 1
            
            # ============= TENSORBOARD: Step metrics =============
            if step % 10 == 0:
                # Box position
                if 'box_x' in info:
                    writer.add_scalar('Step/Box_X', info['box_x'], step)
                    box_position_history.append(info['box_x'])
                
                # Progress
                if 'progress' in info:
                    writer.add_scalar('Step/Progress', info['progress'], step)
                    progress_history.append(info['progress'])
                
                # Active robots
                if 'active_robots' in info:
                    writer.add_scalar('Step/Active_Robots', info['active_robots'], step)
                    active_robots_history.append(info['active_robots'])
                
                # Step reward
                writer.add_scalar('Step/Reward', reward_sum, step)
            
            # Track role weights
            if role_weights is not None and step % 10 == 0:
                if isinstance(role_weights, np.ndarray):
                    if len(role_weights.shape) == 2:
                        avg_roles = np.mean(role_weights, axis=0)
                        role_weight_history.append(avg_roles)
                        
                        # ============= TENSORBOARD: Role weights =============
                        for i, name in enumerate(role_names):
                            if i < len(avg_roles):
                                writer.add_scalar(f'Roles/{name}', avg_roles[i], step)
            
            # Episode end
            if done:
                episode += 1
                episode_rewards.append(episode_reward)
                cooperation_rates.append(info.get('cooperations', 0))
                collision_rates.append(info.get('collisions', 0))
                
                # Check if success
                if terminated:
                    success_count += 1
                    total_successes += 1
                
                # ============= TENSORBOARD: Episode metrics =============
                writer.add_scalar('Episode/Reward', episode_reward, episode)
                writer.add_scalar('Episode/Length', episode_steps, episode)
                writer.add_scalar('Episode/Cooperations', info.get('cooperations', 0), episode)
                writer.add_scalar('Episode/Collisions', info.get('collisions', 0), episode)
                writer.add_scalar('Episode/Box_X', info.get('box_x', 0), episode)
                writer.add_scalar('Episode/Progress', info.get('progress', 0), episode)
                writer.add_scalar('Episode/Active_Robots', info.get('active_robots', 0), episode)
                writer.add_scalar('Episode/Touching_Box', info.get('touching_box', 0), episode)
                writer.add_scalar('Episode/Success', 1 if terminated else 0, episode)
                
                # Success rate (last 10 episodes)
                recent_successes = sum(1 for i in range(max(0, episode-10), episode) 
                                     if i in locals() and terminated)
                writer.add_scalar('Episode/Success_Rate_10ep', recent_successes / 10, episode)
                
                if len(model.buffer) >= model.batch_size:
                    try:
                        next_state = info.get('state')
                        
                        if next_state is None:
                            next_state = np.zeros(model.state_dim)
                        elif isinstance(next_state, np.ndarray):
                            if next_state.shape[0] != model.state_dim:
                                if len(next_state.shape) == 1:
                                    if next_state.shape[0] > model.state_dim:
                                        next_state = next_state[:model.state_dim]
                                    else:
                                        next_state = np.pad(next_state, (0, model.state_dim - next_state.shape[0]))
                                else:
                                    next_state = np.zeros(model.state_dim)
                        else:
                            next_state = np.zeros(model.state_dim)
                        
                        update_info = model.update(next_state, done)
                        
                        # ============= TENSORBOARD: Loss metrics =============
                        writer.add_scalar('Loss/Actor', update_info.get('actor_loss', 0), step)
                        writer.add_scalar('Loss/Critic', update_info.get('critic_loss', 0), step)
                        writer.add_scalar('Loss/Entropy', update_info.get('entropy', 0), step)
                        writer.add_scalar('Loss/Clip_Fraction', update_info.get('clip_fraction', 0), step)
                        
                        if episode % 5 == 0:
                            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
                            avg_coop = np.mean(cooperation_rates) if cooperation_rates else 0
                            avg_coll = np.mean(collision_rates) if collision_rates else 0
                            
                            role_str = ""
                            if len(role_weight_history) > 0:
                                role_list = list(role_weight_history)
                                last_n = min(20, len(role_list))
                                if last_n > 0:
                                    avg_roles = np.mean(role_list[-last_n:], axis=0)
                                    role_str = " | Roles: "
                                    for i, name in enumerate(role_names):
                                        if i < len(avg_roles):
                                            role_str += f"{name}:{avg_roles[i]:.2f} "
                            
                            # Calculate elapsed time and steps per second
                            elapsed_time = time.time() - train_start_time
                            steps_per_sec = step / max(elapsed_time, 1e-6)
                            
                            print(f"\n[Episode {episode:3d}] Step: {step:6d} | "
                                  f"Reward: {avg_reward:7.2f} | Coop: {avg_coop:4.1f} | "
                                  f"Coll: {avg_coll:4.1f} | Loss: {update_info.get('actor_loss', 0):.3f}"
                                  f"{role_str}")
                            print(f"              Box X: {info.get('box_x', 0):6.2f} | "
                                  f"Progress: {info.get('progress', 0):5.2f} | "
                                  f"Success Rate: {success_count/5*100:.1f}% | "
                                  f"Steps/sec: {steps_per_sec:.1f}")
                            
                            # Reset success counter
                            success_count = 0
                            
                            if episode % 20 == 0:
                                model.save_model()
                                # Save TensorBoard metadata
                                writer.add_text('Checkpoint', f'Saved at episode {episode}', episode)
                                
                    except Exception as e:
                        print(f"⚠️ Update error at episode {episode}: {e}")
                        traceback.print_exc()
                else:
                    if episode % 10 == 0:
                        print(f"  Episode {episode:3d} | Filling buffer: {len(model.buffer)}/{model.buffer_size} "
                              f"(need {model.batch_size})")
                
                print(f"  Episode {episode:3d} complete | Reward: {episode_reward:.2f} | Steps: {episode_steps}")
                obs, info = env.reset()
                episode_reward = 0
                episode_steps = 0
            
            # Periodic evaluation
            if step > 0 and step % 10000 == 0:
                print(f"\n>>> Saving checkpoint at step {step}...")
                model.save_model(f"./models/mappo_3algo_{args.bio}/mappo_step_{step}.pt")
                
                # ============= TENSORBOARD: Model checkpoint =============
                writer.add_text('Model', f'Saved checkpoint at step {step}', step)
                
                # Log statistics
                if len(box_position_history) > 0:
                    avg_box_x = np.mean(list(box_position_history)[-100:])
                    writer.add_scalar('Stats/Avg_Box_X', avg_box_x, step)
                
                if len(progress_history) > 0:
                    avg_progress = np.mean(list(progress_history)[-100:])
                    writer.add_scalar('Stats/Avg_Progress', avg_progress, step)
                
                if len(active_robots_history) > 0:
                    avg_active = np.mean(list(active_robots_history)[-100:])
                    writer.add_scalar('Stats/Avg_Active_Robots', avg_active, step)
                
                # Log role weights
                if len(role_weight_history) > 0:
                    role_list = list(role_weight_history)
                    last_n = min(100, len(role_list))
                    if last_n > 0:
                        role_avg = np.mean(role_list[-last_n:], axis=0)
                        print(f">>> Role weights (avg last {last_n}):")
                        for i, name in enumerate(role_names):
                            if i < len(role_avg):
                                print(f"    {name}: {role_avg[i]:.3f}")
                                writer.add_scalar(f'Roles_Avg/{name}', role_avg[i], step)
    
    except KeyboardInterrupt:
        print("\n\n>>> Training interrupted by user")
    except Exception as e:
        print(f"\n>>> Unexpected error: {e}")
        traceback.print_exc()
    
    # ============= FINAL TENSORBOARD LOGGING =============
    # Save final model
    try:
        model.save_model()
        final_model_path = f"./models/mappo_3algo_{args.bio}/final_model.pt"
        model.save_model(final_model_path)
        print(f"\n>>> Final model saved to {final_model_path}")
        writer.add_text('Model', f'Final model saved at step {args.steps}', args.steps)
    except Exception as e:
        print(f"⚠️ Error saving final model: {e}")
    
    # Log final role weights
    if len(role_weight_history) > 0:
        role_list = list(role_weight_history)
        last_n = min(50, len(role_list))
        if last_n > 0:
            final_roles = np.mean(role_list[-last_n:], axis=0)
            print(f"\n>>> FINAL ROLE WEIGHTS (avg last {last_n} episodes):")
            print("=" * 50)
            for i, name in enumerate(role_names):
                if i < len(final_roles):
                    print(f"    {name:10s}: {final_roles[i]:.3f}")
                    writer.add_scalar(f'Final_Roles/{name}', final_roles[i], args.steps)
            print("=" * 50)
    
    # Log training summary
    total_time = time.time() - train_start_time
    writer.add_text('Summary', f'Total training time: {total_time/60:.2f} minutes', 0)
    writer.add_text('Summary', f'Total episodes: {episode}', 0)
    writer.add_text('Summary', f'Total successes: {total_successes}', 0)
    writer.add_text('Summary', f'Success rate: {total_successes/episode*100:.2f}%' if episode > 0 else 'N/A', 0)
    writer.add_text('Summary', f'Final box position: {info.get("box_x", 0):.2f}', 0)
    writer.add_text('Summary', f'Final progress: {info.get("progress", 0):.2f}', 0)
    
    # Close TensorBoard writer
    writer.close()
    print(f"\n>>> TensorBoard logs saved to {log_dir}")
    print(f">>> To view: tensorboard --logdir {args.log_dir}")
    
    try:
        env.close()
        print("[Env] Environment closed")
    except:
        pass
    
    print("\n" + "=" * 70)
    print("🏁 TRAINING COMPLETE - 3 BIO-HYBRID ALGORITHMS")
    print(f"Total Episodes: {episode}")
    print(f"Total Successes: {total_successes}")
    if episode > 0:
        print(f"Success Rate: {total_successes/episode*100:.2f}%")
    if episode_rewards:
        print(f"Average Reward (last 10): {np.mean(list(episode_rewards)[-10:]):.2f}")
    print(f"Total Training Time: {total_time/60:.2f} minutes")
    print("=" * 70)
    print(f"\n📊 To view TensorBoard, run:")
    print(f"    tensorboard --logdir {args.log_dir}")
    print(f"    Then open http://localhost:6006 in your browser")

if __name__ == "__main__":
    main()