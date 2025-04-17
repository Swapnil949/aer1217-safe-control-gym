"""Example utility module.

Please use a file like this one to add extra functions.
"""
import numpy as np
import random
from scipy.interpolate import CubicSpline

def exampleFunction():
    """Example of user-defined function."""
    x = -1
    return x

def generateCircle(radius, start_point, num_points):
    """
    Generates a numpy array of 3D coordinates forming a circle based on a starting point on its edge.
    """
    center = (start_point[0] + radius, start_point[1], start_point[2])
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False) + np.pi
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z = np.full_like(x, center[2])
    return np.column_stack((x, y, z))


def augment_waypoints(gates): 
    """
    Based on yaw angle, add waypoint to front and back of each gate
    """
    
    waypoints = []
    dist = 0.25
    
    for i, gate in enumerate(gates):
        x, y, _, _, _, yaw, _ = gate
        dy = dist * np.cos(yaw)
        dx = dist * np.sin(yaw)
        
        # add waypoint in front of gate
        waypoints.append([x + dx, y + dy, yaw])
        
        # add gate center as waypoint
        waypoints.append([x, y, yaw])
        
        # add waypoint behind gate
        waypoints.append([x - dx, y - dy, yaw])
    
    return np.array(waypoints)

def augment_obstacles(gates):
    """
    Based on yaw angle, add edges of the gates as obstacles
    """
    
    obs = []
    
    for gate in gates:
        x, y, _, _, _, yaw, _ = gate
        
        # calculate the endpoint of the gate edge
        x1 = x + 0.4 * np.cos(yaw)
        y1 = y + 0.4 * np.sin(yaw)
        x2 = x - 0.4 * np.cos(yaw)
        y2 = y - 0.4 * np.sin(yaw)
        
        # Add points to the obstacle list
        obs.append([x1, y1, 0])
        obs.append([x2, y2, 0])
    return np.array(obs)

import numpy as np
import heapq


class Node:
    def __init__(self, position, cost, heuristic, parent=None):
        self.position = position
        self.cost = cost
        self.heuristic = heuristic
        self.parent = parent

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)


class AStarPlanner:
    def __init__(self, obstacles, resolution=0.1, obstacle_radius=0.25):
        self.resolution = resolution
        self.obstacle_radius = obstacle_radius
        self.obstacles = obstacles

    def is_collision(self, point):
        for obs in self.obstacles:
            if np.linalg.norm(point - obs[:2]) <= self.obstacle_radius:
                return True
        return False

    def get_neighbors(self, point):
        directions = np.array([
            [1, 0], [-1, 0], [0, 1], [0, -1],
            [1, 1], [1, -1], [-1, 1], [-1, -1]
        ])
        neighbors = []
        for d in directions:
            neighbor = point + self.resolution * d
            if not self.is_collision(neighbor):
                neighbors.append(neighbor)
        return neighbors

    def heuristic(self, a, b):
        return np.linalg.norm(a - b)

    def plan(self, start, goal):
        start = np.round(start / self.resolution) * self.resolution
        goal = np.round(goal / self.resolution) * self.resolution

        open_list = []
        heapq.heappush(open_list, Node(start, 0.0, self.heuristic(start, goal)))
        visited = {}

        while open_list:
            current = heapq.heappop(open_list)
            c_pos = tuple(current.position.round(2))

            if c_pos in visited:
                continue
            visited[c_pos] = current

            if self.heuristic(current.position, goal) <= self.resolution:
                return self.reconstruct_path(current)

            for neighbor in self.get_neighbors(current.position):
                n_pos = tuple(neighbor.round(2))
                if n_pos in visited:
                    continue
                cost = current.cost + np.linalg.norm(neighbor - current.position)
                heuristic = self.heuristic(neighbor, goal)
                heapq.heappush(open_list, Node(neighbor, cost, heuristic, current))

        return None  # No path found

    def reconstruct_path(self, node):
        path = []
        while node:
            path.append(node.position)
            node = node.parent
        return path[::-1]

    def plan_through_waypoints(self, start, waypoints):
        full_path = []
        current_start = start

        for wp in waypoints:
            path_segment = self.plan(current_start, wp[:2])
            if path_segment is None:
                print(f"Failed to reach waypoint {wp[:2]}")
                return None
            if full_path:
                full_path.extend(path_segment[1:])  # skip duplicate
            else:
                full_path.extend(path_segment)
            current_start = wp[:2]

        return np.array(full_path)

def smooth_path_with_cubic_spline(path, num_points=300):
    """
    Smoothens a 3D path using cubic splines with arc-length parameterization.

    Args:
        path (np.ndarray): N x 3 array of waypoints (x, y, z).
        num_points (int): Number of points to interpolate for the smooth path.

    Returns:
        np.ndarray: Smoothed path with shape (num_points, 3)
    """
    if path.shape[0] < 2:
        raise ValueError("Path must have at least 2 points to smooth.")

    # Compute arc-length parameter (cumulative distance)
    distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate(([0], np.cumsum(distances)))

    # Fit separate cubic splines for x(s), y(s), z(s)
    cs_x = CubicSpline(s, path[:, 0])
    cs_y = CubicSpline(s, path[:, 1])
    #cs_z = CubicSpline(s, path[:, 2])

    # Resample along the arc-length
    s_new = np.linspace(0, s[-1], num_points)
    x_new = cs_x(s_new)
    y_new = cs_y(s_new)
    # z_new = cs_z(s_new)

    smoothed_path = np.vstack((x_new, y_new)).T
    return smoothed_path

def get_path(start, goal, obstacles, gates):
    obstacles = [o[:3] for o in obstacles]
    waypoints = augment_waypoints(gates)
    obs = augment_obstacles(gates)
    obstacles = np.vstack((obstacles, obs))
    planner = AStarPlanner(obstacles, resolution=0.1, obstacle_radius=0.4)
    
    waypoints = np.vstack((start[:2], waypoints[:, :2], goal[:2]))
    path = planner.plan_through_waypoints(start[:2], waypoints)
    if path is None:
        print("No path found")
        return None
    
    # Smooth the path
    path = smooth_path_with_cubic_spline(path, num_points=550)
    
    
    # Add 1.0 as z coordinate to the path
    path = np.hstack((path, np.ones((path.shape[0], 1))))
    
    return path 