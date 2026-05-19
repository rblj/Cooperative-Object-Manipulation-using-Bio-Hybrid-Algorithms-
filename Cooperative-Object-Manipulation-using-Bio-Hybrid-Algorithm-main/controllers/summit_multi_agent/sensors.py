import math
import numpy as np
from controller import GPS, Compass, Lidar

class SensorSuite:
    def __init__(self, robot, timestep):
        self.robot = robot

        # --- GPS ---
        self.gps = robot.getDevice("gps")
        if self.gps:
            self.gps.enable(timestep)
        else:
            print(f"[Warning] GPS not found on {robot.getName()}")

        # --- Compass ---
        self.compass = robot.getDevice("compass")
        if self.compass:
            self.compass.enable(timestep)
        else:
            print(f"[Warning] Compass not found on {robot.getName()}")

        # --- LiDAR ---
        # Adjust the name here if your LiDAR is named differently in Webots
        lidar_names = ["lidar", "hokuyo", "Lidar"]  # try multiple common names
        self.lidar = None
        for name in lidar_names:
            device = robot.getDevice(name)
            if device:
                self.lidar = device
                break

        if self.lidar:
            self.lidar.enable(timestep)
            self.lidar.enablePointCloud()
        else:
            print(f"[Warning] LiDAR not found on {robot.getName()}, obstacle vector will be zero.")

    # --- Robot global pose ---
    def get_world_state(self):
        x, z = 0.0, 0.0
        heading = 0.0

        if self.gps:
            pos = self.gps.getValues()
            x = pos[0]
            z = pos[2]

        if self.compass:
            north = self.compass.getValues()
            heading = math.atan2(north[0], north[2])  # radians

        return x, z, heading

    # --- Obstacle vector (closest point) ---
    def get_obstacle_vector(self):
        if self.lidar is None:
            return 0.0, 0.0

        ranges = self.lidar.getRangeImage()
        if not ranges:
            return 0.0, 0.0

        min_dist = min(ranges)
        idx = ranges.index(min_dist)
        angle = (idx / len(ranges)) * 2 * math.pi - math.pi

        obs_x = min_dist * math.cos(angle)
        obs_z = min_dist * math.sin(angle)

        return obs_x, obs_z

    # --- Dummy box info (RL learns via reward, not vision) ---
    def get_box_data(self):
        return {
            "found": True,
            "dist": 1.0
        }
