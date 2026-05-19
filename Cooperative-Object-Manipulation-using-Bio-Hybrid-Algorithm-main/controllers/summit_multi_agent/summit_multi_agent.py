import socket
import json
import numpy as np
from controller import Robot, GPS, Compass, Lidar
import time
import math
import select
from collections import deque

# Initialize robot
robot = Robot()
timestep = int(robot.getBasicTimeStep())
robot_name = robot.getName()

print(f"[{robot_name}] Initializing Summit XL for MAPPO with 3 bio-hybrid algorithms...")

# Setup wheels
wheel_names = ["front_left_wheel_joint", "front_right_wheel_joint",
               "back_left_wheel_joint", "back_right_wheel_joint"]
wheels = []
for name in wheel_names:
    try:
        wheel = robot.getDevice(name)
        wheel.setPosition(float('inf'))
        wheel.setVelocity(0.0)
        wheels.append(wheel)
    except:
        print(f"[{robot_name}] Could not find wheel: {name}")

# Global state
current_action = [0.0, 0.0]
current_role_weights = [0.33, 0.33, 0.34]
is_connected = False
connection = None
last_connect_time = 0
start_time = time.time()

# Robot state
robot_pos = [0.0, 0.0]
robot_vel = [0.0, 0.0]
box_position = [12.0, 0.0]
pheromone_strength = 0.0
pheromone_gradient = [0.0, 0.0, 0.0]
visited_intensity = 0.0
neighbors = []
time_elapsed = 0.0

# Bio-hybrid signals for 3 algorithms
global_best_position = [12.0, 0.0]
personal_best = [0.0, 0.0]
pso_velocity = [0.0, 0.0]
flock_center = [0.0, 0.0]
flock_velocity = [0.0, 0.0]

# Sensors
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# ============= LIDAR SETUP =============
lidar = None
has_lidar = False
lidar_max_range = 10.0
lidar_horizontal_res = 0

try:
    lidar = robot.getDevice("frontLidar")
    if lidar is not None:
        lidar.enable(timestep)
        lidar.enablePointCloud()
        lidar_horizontal_res = lidar.getHorizontalResolution()
        lidar_max_range = lidar.getMaxRange()
        print(f"[{robot_name}] ✅ Front LiDAR enabled: {lidar_horizontal_res} points, max range: {lidar_max_range:.1f}m")
        has_lidar = True
except Exception as e:
    print(f"[{robot_name}] Front LiDAR not found: {e}")
    
    try:
        lidar = robot.getDevice("backLidar")
        if lidar is not None:
            lidar.enable(timestep)
            lidar.enablePointCloud()
            lidar_horizontal_res = lidar.getHorizontalResolution()
            lidar_max_range = lidar.getMaxRange()
            print(f"[{robot_name}] ✅ Back LiDAR enabled: {lidar_horizontal_res} points, max range: {lidar_max_range:.1f}m")
            has_lidar = True
    except Exception as e:
        print(f"[{robot_name}] Back LiDAR not found: {e}")

if not has_lidar:
    print(f"[{robot_name}] ⚠️ No LiDAR found, using simulated sensor")
    lidar_max_range = 10.0

# Connection parameters
SERVER_HOST = "localhost"
SERVER_PORT = 12345
RECONNECT_DELAY = 2.0

# Bio-hybrid memory
visited_positions = deque(maxlen=100)
pheromone_trail = deque(maxlen=100)

# ============= OBJECT IDENTIFICATION FUNCTION =============
def identify_object(robot_pos, distance, angle, box_pos, neighbors, robot_name):
    """
    Identify what object the robot is seeing:
    Returns: 'box', 'robot', 'wall', or 'unknown'
    """
    # ============= 1. CHECK IF IT'S THE BOX =============
    dx = box_pos[0] - robot_pos[0]
    dy = box_pos[1] - robot_pos[1]
    expected_box_distance = math.sqrt(dx*dx + dy*dy)
    expected_box_angle = math.atan2(dy, dx)
    if expected_box_angle < 0:
        expected_box_angle += 2 * math.pi
    
    if abs(distance - expected_box_distance) < 0.5 and abs(angle - expected_box_angle) < 0.3:
        return 'box'
    
    # ============= 2. CHECK IF IT'S ANOTHER ROBOT =============
    for neighbor in neighbors:
        neighbor_x = neighbor.get('x', 0)
        neighbor_y = neighbor.get('y', 0)
        neighbor_dist = math.sqrt((neighbor_x - robot_pos[0])**2 + (neighbor_y - robot_pos[1])**2)
        neighbor_angle = math.atan2(neighbor_y - robot_pos[1], neighbor_x - robot_pos[0])
        if neighbor_angle < 0:
            neighbor_angle += 2 * math.pi
        
        if abs(distance - neighbor_dist) < 0.3 and abs(angle - neighbor_angle) < 0.2:
            return 'robot'
    
    # ============= 3. CHECK IF IT'S A WALL =============
    ray_x = robot_pos[0] + math.cos(angle) * distance
    ray_y = robot_pos[1] + math.sin(angle) * distance
    
    if abs(ray_x) > 14.5 or abs(ray_y) > 9.5:
        return 'wall'
    
    return 'unknown'

# ============= ENHANCE BOX IN LIDAR =============
def enhance_box_in_lidar(range_image, robot_pos, box_pos):
    """Make the box VISIBLE and CLEAR in LiDAR scans"""
    if range_image is None or len(range_image) == 0:
        return range_image
    
    try:
        filtered = np.array(range_image, dtype=np.float32)
        
        dx = box_pos[0] - robot_pos[0]
        dy = box_pos[1] - robot_pos[1]
        distance_to_box = math.sqrt(dx*dx + dy*dy)
        
        if distance_to_box < 10.0:
            angle_to_box = math.atan2(dy, dx)
            if angle_to_box < 0:
                angle_to_box += 2 * math.pi
            
            num_points = len(filtered)
            for i in range(num_points):
                point_angle = i * (2 * math.pi / num_points)
                angle_diff = abs(point_angle - angle_to_box)
                if angle_diff > math.pi:
                    angle_diff = 2 * math.pi - angle_diff
                
                if angle_diff < 0.3:
                    filtered[i] = min(distance_to_box, lidar_max_range)
        
        return filtered.tolist()
    except Exception as e:
        print(f"[{robot_name}] Enhance error: {e}")
        return range_image

# ============= GET ENHANCED LIDAR SCAN =============
def get_enhanced_lidar_scan(robot_pos, box_pos, neighbors):
    """Get LiDAR scan with object identification for each sector"""
    
    if not has_lidar or lidar is None:
        # Simulated LiDAR
        distances = np.ones(8, dtype=np.float32) * lidar_max_range
        types = ['unknown'] * 8
        
        for i in range(8):
            angle = i * (2 * math.pi / 8)
            ray_x = robot_pos[0] + math.cos(angle) * 2.0
            ray_y = robot_pos[1] + math.sin(angle) * 2.0
            
            if abs(ray_x) > 14.0 or abs(ray_y) > 8.0:
                distances[i] = 2.0
                types[i] = 'wall'
            else:
                dist_to_box = ray_box_intersection(robot_pos, angle, box_position)
                if dist_to_box < 5.0:
                    distances[i] = dist_to_box
                    types[i] = 'box'
        
        return {
            'distances': distances,
            'types': types
        }
    
    try:
        range_image = lidar.getRangeImage()
        if range_image is None or len(range_image) == 0:
            return {
                'distances': np.ones(8, dtype=np.float32) * lidar_max_range,
                'types': ['unknown'] * 8
            }
        
        if not isinstance(range_image, list):
            range_image = list(range_image)
        
        # Enhance the box
        enhanced_image = enhance_box_in_lidar(range_image, robot_pos, box_pos)
        
        num_points = len(enhanced_image)
        sector_distances = []
        sector_types = []
        
        for i in range(8):
            start_idx = i * (num_points // 8)
            end_idx = min(start_idx + (num_points // 8), num_points)
            
            if start_idx >= num_points:
                sector_distances.append(lidar_max_range)
                sector_types.append('unknown')
                continue
            
            sector_vals = enhanced_image[start_idx:end_idx]
            sector_vals = np.array(sector_vals, dtype=np.float32)
            
            valid_mask = (~np.isinf(sector_vals)) & (~np.isnan(sector_vals))
            valid_vals = sector_vals[valid_mask]
            
            if len(valid_vals) > 0:
                min_dist = np.min(valid_vals)
                sector_distances.append(min_dist)
                
                # Identify object
                sector_angle = i * (2 * math.pi / 8)
                object_type = identify_object(
                    robot_pos, min_dist, sector_angle, 
                    box_pos, neighbors, robot_name
                )
                sector_types.append(object_type)
            else:
                sector_distances.append(lidar_max_range)
                sector_types.append('nothing')
        
        return {
            'distances': np.array(sector_distances[:8], dtype=np.float32),
            'types': sector_types[:8]
        }
        
    except Exception as e:
        return {
            'distances': np.ones(8, dtype=np.float32) * lidar_max_range,
            'types': ['unknown'] * 8
        }

def ray_box_intersection(ray_start, angle, box_pos):
    """Check if ray hits box"""
    ray_dir = np.array([math.cos(angle), math.sin(angle)])
    box_half_width = 1.0
    box_half_height = 0.75
    
    t_near = -float('inf')
    t_far = float('inf')
    
    for i in range(2):
        if abs(ray_dir[i]) < 1e-6:
            if ray_start[i] < box_pos[i] - box_half_width or ray_start[i] > box_pos[i] + box_half_width:
                return float('inf')
        else:
            t1 = (box_pos[i] - box_half_width - ray_start[i]) / ray_dir[i]
            t2 = (box_pos[i] + box_half_width - ray_start[i]) / ray_dir[i]
            
            if t1 > t2:
                t1, t2 = t2, t1
            
            if t1 > t_near:
                t_near = t1
            if t2 < t_far:
                t_far = t2
            
            if t_near > t_far or t_far < 0:
                return float('inf')
    
    return t_near if t_near > 0 else float('inf')

# ============= GET ROBOT STATE =============
def get_robot_state():
    """Get complete robot state for RL with enhanced LiDAR"""
    try:
        global time_elapsed
        time_elapsed = time.time() - start_time
        
        pos = gps.getValues()
        robot_pos[0], robot_pos[1] = pos[0], pos[1]
        
        if hasattr(get_robot_state, 'last_pos'):
            robot_vel[0] = robot_pos[0] - get_robot_state.last_pos[0]
            robot_vel[1] = robot_pos[1] - get_robot_state.last_pos[1]
        get_robot_state.last_pos = robot_pos.copy()
        
        try:
            compass_vals = compass.getValues()
            orientation = math.atan2(compass_vals[0], compass_vals[1])
        except:
            orientation = 0.0
        
        # Get enhanced LiDAR with object types
        lidar_data = get_enhanced_lidar_scan(robot_pos, box_position, neighbors)
        lidar_distances = lidar_data['distances']
        lidar_types = lidar_data['types']
        
        visited_positions.append((robot_pos[0], robot_pos[1]))
        
        return {
            'position': robot_pos.copy(),
            'velocity': robot_vel.copy(),
            'orientation': orientation,
            'lidar': lidar_distances.tolist(),
            'lidar_types': lidar_types,
            'box_position': box_position,
            'pheromone': float(pheromone_strength),
            'pheromone_gradient': pheromone_gradient,
            'visited_intensity': visited_intensity,
            'neighbors': neighbors,
            'time': time_elapsed,
            'global_best_position': global_best_position,
            'personal_best': personal_best,
            'pso_velocity': pso_velocity,
            'flock_center': flock_center,
            'flock_velocity': flock_velocity,
        }
    except Exception as e:
        return {
            'position': [0.0, 0.0],
            'velocity': [0.0, 0.0],
            'orientation': 0.0,
            'lidar': [lidar_max_range] * 8,
            'lidar_types': ['unknown'] * 8,
            'box_position': box_position,
            'pheromone': 0.0,
            'pheromone_gradient': [0.0, 0.0, 0.0],
            'visited_intensity': 0.0,
            'neighbors': [],
            'time': 0.0,
            'global_best_position': [12.0, 0.0],
            'personal_best': [0.0, 0.0],
            'pso_velocity': [0.0, 0.0],
            'flock_center': [0.0, 0.0],
            'flock_velocity': [0.0, 0.0],
        }

# ============= APPLY BEHAVIORS =============
def apply_bio_hybrid_behaviors(action, state, role_weights):
    """
    CLEAN, SIMPLE, WORKING BEHAVIOR:
    1. 🚧 OBSTACLE - Avoid walls, cooperate with robots
    2. 🎯 GOAL - Go to box (STRONG)
    3. 📦 PUSH - Push box toward goal (MAX POWER)
    4. 🤝 COOPERATE - Help others push
    5. 🧭 EXPLORE - Only when lost
    """
    linear, angular = action
    
    # ============= GET STATE =============
    robot_x, robot_y = state.get('position', [0.0, 0.0])
    box_x, box_y = state.get('box_position', [12.0, 0.0])
    goal_x, goal_y = -12.0, 0.0
    current_angle = state.get('orientation', 0.0)
    
    # Get enhanced LiDAR with object types
    lidar_distances = np.array(state.get('lidar', [10.0]*8))
    lidar_types = state.get('lidar_types', ['unknown']*8)
    
    neighbors = state.get('neighbors', [])
    
    dist_to_box = math.sqrt((box_x - robot_x)**2 + (box_y - robot_y)**2)
    dist_to_goal = math.sqrt((goal_x - box_x)**2 + (goal_y - box_y)**2)
    box_velocity_x = state.get('box_velocity', [0.0, 0.0])[0]
    
    # ============= GET ROLE WEIGHTS =============
    ant_weight = role_weights[0] if len(role_weights) >= 1 else 0.0
    pso_weight = role_weights[1] if len(role_weights) >= 2 else 0.0
    flock_weight = role_weights[2] if len(role_weights) >= 3 else 0.0
    
    # ============= 1. 🚧 SMART OBSTACLE AVOIDANCE =============
    if len(lidar_distances) >= 8:
        front_indices = [0, 1, 7]
        front_distances = [lidar_distances[i] for i in front_indices]
        front_types = [lidar_types[i] for i in front_indices]
        
        min_front = min(front_distances)
        min_front_idx = front_distances.index(min_front)
        front_object_type = front_types[min_front_idx]
        
        if min_front < 1.0:
            if front_object_type == 'box':
                # ✅ IT'S THE BOX! GO TOWARD IT!
                target_angle = math.atan2(box_y - robot_y, box_x - robot_x)
                angle_diff = target_angle - current_angle
                while angle_diff > math.pi: angle_diff -= 2*math.pi
                while angle_diff < -math.pi: angle_diff += 2*math.pi
                angular += angle_diff * 2.0
                linear += 0.9
                
            elif front_object_type == 'robot':
                # 🤝 ANOTHER ROBOT! COOPERATE!
                for neighbor in neighbors:
                    neighbor_dist = neighbor.get('distance', 10.0)
                    if abs(neighbor_dist - min_front) < 0.3:
                        if neighbor.get('dist_to_box', 10.0) < dist_to_box:
                            linear *= 0.3
                            if lidar_distances[0] < lidar_distances[7]:
                                angular -= 0.2
                            else:
                                angular += 0.2
                        else:
                            linear *= 0.8
                        break
                        
            elif front_object_type == 'wall':
                # 🚧 WALL! AVOID IMMEDIATELY!
                linear = -0.3
                if lidar_distances[0] < lidar_distances[7]:
                    angular = 0.8
                else:
                    angular = -0.8
                return np.clip([linear, angular], -1, 1)
    
    # ============= 2. 🎯 GO TO BOX =============
    if dist_to_box > 0.8:
        target_angle = math.atan2(box_y - robot_y, box_x - robot_x)
        angle_diff = target_angle - current_angle
        while angle_diff > math.pi: angle_diff -= 2*math.pi
        while angle_diff < -math.pi: angle_diff += 2*math.pi
        
        angular += angle_diff * 1.5
        linear += 0.8 * min(1.0, dist_to_box / 4.0)
        print(f"[{robot_name}] 🎯 Going to box: {dist_to_box:.2f}m")
    
    # ============= 3. 📦 PUSH BOX =============
    if dist_to_box <= 0.8:
        push_dir_x = goal_x - box_x
        push_dir_y = goal_y - box_y
        push_dist = math.sqrt(push_dir_x**2 + push_dir_y**2)
        
        if push_dist > 0:
            push_dir_x /= push_dist
            push_dir_y /= push_dist
            
            target_angle = math.atan2(push_dir_y, push_dir_x)
            angle_diff = target_angle - current_angle
            while angle_diff > math.pi: angle_diff -= 2*math.pi
            while angle_diff < -math.pi: angle_diff += 2*math.pi
            
            angular += angle_diff * 1.0
            linear += 1.0
            print(f"[{robot_name}] 📦 PUSHING BOX toward goal!")
    
    # ============= 4. 🤝 COOPERATION =============
    if len(neighbors) > 0:
        robots_at_box = 0
        for neighbor in neighbors:
            if neighbor.get('dist_to_box', 10.0) < 1.0:
                robots_at_box += 1
        
        if dist_to_box > 1.5 and robots_at_box > 0:
            target_angle = math.atan2(box_y - robot_y, box_x - robot_x)
            angle_diff = target_angle - current_angle
            while angle_diff > math.pi: angle_diff -= 2*math.pi
            while angle_diff < -math.pi: angle_diff += 2*math.pi
            
            angular += angle_diff * 1.2
            linear += 0.7
            print(f"[{robot_name}] 🤝 Helping! Robots at box: {robots_at_box}")
        
        elif dist_to_box < 1.0 and robots_at_box >= 1:
            push_dir_x = goal_x - box_x
            push_dir_y = goal_y - box_y
            push_dist = math.sqrt(push_dir_x**2 + push_dir_y**2)
            
            if push_dist > 0:
                push_dir_x /= push_dist
                push_dir_y /= push_dist
                
                target_angle = math.atan2(push_dir_y, push_dir_x)
                angle_diff = target_angle - current_angle
                while angle_diff > math.pi: angle_diff -= 2*math.pi
                while angle_diff < -math.pi: angle_diff += 2*math.pi
                
                angular += angle_diff * 0.6
                linear += 1.0
                print(f"[{robot_name}] 🚂 TEAM PUSH! Robots: {robots_at_box + 1}")
    
    # ============= 5. 🧬 BIO-HYBRID MODULATION =============
    if len(role_weights) >= 3:
        # 🐜 ANT COLONY
        if ant_weight > 0.2:
            gradient = state.get('pheromone_gradient', [0.0, 0.0, 0.0])
            if gradient[1] > gradient[2]:
                angular -= ant_weight * 0.2
            else:
                angular += ant_weight * 0.2
            if gradient[0] > 0.5:
                linear += ant_weight * 0.2
        
        # 🐝 PARTICLE SWARM
        if pso_weight > 0.2:
            global_best = state.get('global_best_position', [12.0, 0.0])
            dx = global_best[0] - robot_x
            dy = global_best[1] - robot_y
            if math.sqrt(dx*dx + dy*dy) > 0.5:
                target_angle = math.atan2(dy, dx)
                angle_diff = target_angle - current_angle
                while angle_diff > math.pi: angle_diff -= 2*math.pi
                while angle_diff < -math.pi: angle_diff += 2*math.pi
                angular += angle_diff * pso_weight * 0.3
        
        # 🦅 FLOCKING
        if flock_weight > 0.2 and dist_to_box > 4.0:
            flock_center = state.get('flock_center', None)
            if flock_center and len(flock_center) >= 2:
                dx = flock_center[0] - robot_x
                dy = flock_center[1] - robot_y
                if math.sqrt(dx*dx + dy*dy) > 1.5:
                    target_angle = math.atan2(dy, dx)
                    angle_diff = target_angle - current_angle
                    while angle_diff > math.pi: angle_diff -= 2*math.pi
                    while angle_diff < -math.pi: angle_diff += 2*math.pi
                    angular += angle_diff * flock_weight * 0.2
    
    # ============= 6. 🧭 EXPLORATION =============
    if dist_to_box > 15.0 and len(neighbors) == 0:
        time_val = state.get('time', 0.0)
        spiral_angle = time_val * 0.2
        angular += math.sin(spiral_angle) * 0.3
        linear += 0.3
        print(f"[{robot_name}] 🧭 Searching for box...")
    
    # ============= 7. 🏆 SUCCESS =============
    if dist_to_goal < 2.0:
        print(f"[{robot_name}] 🏆 SUCCESS! Box reached goal!")
        linear = 0.0
        angular = 0.0
    
    return np.clip([linear, angular], -1, 1)

def apply_action(action):
    """Apply action to robot wheels"""
    if action is None:
        for wheel in wheels:
            wheel.setVelocity(0.0)
        return
    
    linear, angular = action
    
    base_speed = 4.0
    left_speed = (linear - angular) * base_speed
    right_speed = (linear + angular) * base_speed
    
    max_speed = 6.0
    left_speed = np.clip(left_speed, -max_speed, max_speed)
    right_speed = np.clip(right_speed, -max_speed, max_speed)
    
    if len(wheels) >= 4:
        wheels[0].setVelocity(left_speed)
        wheels[2].setVelocity(left_speed)
        wheels[1].setVelocity(right_speed)
        wheels[3].setVelocity(right_speed)

# ============= SERVER COMMUNICATION =============
def connect_to_server():
    """Connect to RL training server"""
    global connection, is_connected
    
    try:
        print(f"[{robot_name}] Connecting to {SERVER_HOST}:{SERVER_PORT}...")
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((SERVER_HOST, SERVER_PORT))
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        
        info = {
            'name': robot_name,
            'has_lidar': has_lidar,
            'lidar_type': 'frontLidar' if has_lidar and 'front' in str(lidar) else 'backLidar'
        }
        sock.sendall(json.dumps(info).encode() + b'\n')
        
        try:
            data = sock.recv(256).decode().strip()
            print(f"[{robot_name}] Server ACK: {data}")
        except:
            pass
        
        sock.setblocking(False)
        connection = sock
        is_connected = True
        
        print(f"[{robot_name}] Connected successfully!")
        return True
        
    except Exception as e:
        print(f"[{robot_name}] Connection failed: {e}")
        return False

def send_state(sock):
    """Send robot state to server"""
    try:
        state = get_robot_state()
        message = json.dumps(state) + '\n'
        sock.sendall(message.encode())
        return True
    except (ConnectionResetError, BrokenPipeError):
        return False
    except Exception as e:
        return False

def receive_messages(sock):
    """Receive messages from server"""
    try:
        ready, _, _ = select.select([sock], [], [], 0.01)
        if not ready:
            return None, None, None
        
        data = b""
        try:
            sock.settimeout(0.1)
            chunk = sock.recv(4096)
            if not chunk:
                return 'disconnect', None, None
            data = chunk
            
            try:
                while True:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    data += chunk
            except (socket.timeout, BlockingIOError):
                pass
            sock.settimeout(None)
                
        except BlockingIOError:
            return None, None, None
        except ConnectionResetError:
            return 'disconnect', None, None
        except socket.timeout:
            pass
        
        if not data:
            return None, None, None
        
        data_str = data.decode('utf-8', errors='ignore').strip()
        
        try:
            msg = json.loads(data_str)
            
            if 'action' in msg:
                return 'action', msg['action'], None
            if 'role_weights' in msg:
                return 'role_weights', None, msg['role_weights']
            if 'box_position' in msg:
                global box_position
                box_position = msg['box_position']
                return 'box_update', box_position, None
            if 'pheromone' in msg:
                global pheromone_strength
                pheromone_strength = msg['pheromone']
                return 'pheromone', pheromone_strength, None
            if 'pheromone_gradient' in msg:
                global pheromone_gradient
                pheromone_gradient = msg['pheromone_gradient']
                return 'gradient', pheromone_gradient, None
            if 'global_best_position' in msg:
                global global_best_position
                global_best_position = msg['global_best_position']
                return 'global_best', global_best_position, None
            if 'personal_best' in msg:
                global personal_best
                personal_best = msg['personal_best']
                return 'personal_best', personal_best, None
            if 'flock_center' in msg:
                global flock_center
                flock_center = msg['flock_center']
                return 'flock_center', flock_center, None
            if 'flock_velocity' in msg:
                global flock_velocity
                flock_velocity = msg['flock_velocity']
                return 'flock_velocity', flock_velocity, None
            if 'neighbors' in msg:
                global neighbors
                neighbors = msg['neighbors']
                return 'neighbors', neighbors, None
            if 'visited_intensity' in msg:
                global visited_intensity
                visited_intensity = msg['visited_intensity']
                return 'visited', visited_intensity, None
            
        except json.JSONDecodeError:
            if data_str == "STOP":
                return 'stop', None, None
        
        return None, None, None
        
    except Exception as e:
        return None, None, None

# ============= MAIN LOOP =============
print(f"[{robot_name}] Ready! Waiting for server...")
print(f"[{robot_name}] Mode: 3 BIO-HYBRID BEHAVIORS")
print(f"[{robot_name}] 🐜 Ant | 🐝 PSO | 🦅 Flock")

position_send_interval = 5
position_counter = 0

while robot.step(timestep) != -1:
    current_time = time.time()
    
    robot_state = get_robot_state()
    
    if not is_connected:
        if current_time - last_connect_time > RECONNECT_DELAY:
            last_connect_time = current_time
            if connect_to_server():
                continue
        
        autonomous_action = [0.2, 0.0]
        default_weights = [0.33, 0.33, 0.34]
        modified_action = apply_bio_hybrid_behaviors(autonomous_action, robot_state, default_weights)
        apply_action(modified_action)
        continue
    
    try:
        position_counter += 1
        if position_counter >= position_send_interval:
            position_counter = 0
            if not send_state(connection):
                is_connected = False
                continue
        
        msg_type, msg_data, role_data = receive_messages(connection)
        
        if msg_type == 'stop':
            apply_action([0.0, 0.0])
            break
        elif msg_type == 'disconnect':
            is_connected = False
            continue
        elif msg_type == 'action':
            current_action = msg_data
        elif msg_type == 'role_weights':
            if len(msg_data) >= 3:
                current_role_weights = msg_data
            else:
                current_role_weights = list(msg_data) + [0.33] * (3 - len(msg_data))
        elif msg_type == 'box_update':
            box_position = msg_data
        elif msg_type in ['pheromone', 'gradient', 'global_best', 'personal_best', 
                         'flock_center', 'flock_velocity', 'neighbors', 'visited']:
            pass
        
        modified_action = apply_bio_hybrid_behaviors(
            current_action, 
            robot_state,
            current_role_weights
        )
        apply_action(modified_action)
        
    except Exception as e:
        print(f"[{robot_name}] Error: {e}")
        is_connected = False
        apply_action([0.0, 0.0])

# Cleanup
print(f"[{robot_name}] Shutting down...")
apply_action([0.0, 0.0])

if connection:
    try:
        connection.close()
    except:
        pass