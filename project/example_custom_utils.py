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
    
class RRTStarPlanner:
    def __init__(self, obstacles, x_limits, y_limits, resolution=0.1, obstacle_radius=0.25, max_iter=1000, step_size=0.5):
        self.obstacles = np.array(obstacles)
        self.x_limits = x_limits
        self.y_limits = y_limits
        self.resolution = resolution
        self.obstacle_radius = obstacle_radius
        self.max_iter = max_iter
        self.step_size = step_size
        self.nodes = []

    class Node:
        def __init__(self, position):
            self.position = np.array(position)
            self.parent = None
            self.cost = 0.0

    def is_collision(self, p1, p2=None):
        if p2 is None:
            for obs in self.obstacles:
                if np.linalg.norm(p1[:2] - obs[:2]) <= self.obstacle_radius:
                    return True
            return False

        steps = int(np.linalg.norm(p2 - p1) / self.resolution)
        for i in range(steps + 1):
            point = p1 + i / max(1, steps) * (p2 - p1)
            if self.is_collision(point):
                return True
        return False

    def get_nearest_node(self, position):
        return min(self.nodes, key=lambda node: np.linalg.norm(node.position - position))

    def get_near_nodes(self, new_node, radius=1.0):
        return [node for node in self.nodes if np.linalg.norm(node.position - new_node.position) <= radius]

    def rewire(self, new_node, near_nodes):
        for node in near_nodes:
            new_cost = new_node.cost + np.linalg.norm(new_node.position - node.position)
            if new_cost < node.cost and not self.is_collision(new_node.position, node.position):
                node.parent = new_node
                node.cost = new_cost

    def extract_path(self, node):
        path = []
        while node:
            path.append(node.position)
            node = node.parent
        return path[::-1]

    def plan(self, start, goal):
        self.nodes = [self.Node(start)]
        for _ in range(self.max_iter):
            rand_point = np.array([
                np.random.uniform(self.x_limits[0], self.x_limits[1]),
                np.random.uniform(self.y_limits[0], self.y_limits[1])
            ])

            nearest_node = self.get_nearest_node(rand_point)
            direction = rand_point - nearest_node.position
            distance = np.linalg.norm(direction)
            if distance > self.step_size:
                direction = direction / distance * self.step_size
            new_position = nearest_node.position + direction

            if self.is_collision(nearest_node.position, new_position):
                continue

            new_node = self.Node(new_position)
            new_node.parent = nearest_node
            new_node.cost = nearest_node.cost + np.linalg.norm(new_position - nearest_node.position)

            near_nodes = self.get_near_nodes(new_node)
            for node in near_nodes:
                potential_cost = node.cost + np.linalg.norm(node.position - new_node.position)
                if potential_cost < new_node.cost and not self.is_collision(node.position, new_node.position):
                    new_node.parent = node
                    new_node.cost = potential_cost

            self.nodes.append(new_node)
            self.rewire(new_node, near_nodes)

            if np.linalg.norm(new_position - goal) < self.step_size and not self.is_collision(new_position, goal):
                goal_node = self.Node(goal)
                goal_node.parent = new_node
                goal_node.cost = new_node.cost + np.linalg.norm(goal - new_position)
                return self.extract_path(goal_node)

        return None

    def plan_through_waypoints(self, start, waypoints):
        full_path = []
        current_start = start

        for wp in waypoints:
            path_segment = self.plan(current_start, wp[:2])
            if path_segment is None:
                print(f"Failed to reach waypoint {wp[:2]}")
                return None
            if full_path:
                full_path.extend(path_segment[1:])
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

def bezier_interp(p0, p1, p2, t):
    """Quadratic Bezier interpolation between p0, p1, p2."""
    return (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2

def smooth_path_with_bezier(waypoints, num_points=300, tightness=0.01):
    """
    Smooths a path using piecewise quadratic Bezier curves.

    Args:
        waypoints (np.ndarray): N x D array of waypoints (2D or 3D).
        num_points (int): Total number of points in the final smooth path.
        tightness (float): Controls how far the control point is from the line.

    Returns:
        np.ndarray: Smoothed path with shape (num_points, D)
    """
    waypoints = np.array(waypoints)
    n_segments = len(waypoints) - 1
    points_per_segment = num_points // n_segments
    dim = waypoints.shape[1]
    
    smoothed = []
    
    for i in range(n_segments):
        p0 = waypoints[i]
        p2 = waypoints[i+1]
        
        # Direction vector and control point
        dir_vec = p2 - p0
        p1 = p0 + tightness * dir_vec  # Control point placed between
        
        t_values = np.linspace(0, 1, points_per_segment, endpoint=False)
        for t in t_values:
            pt = bezier_interp(p0, p1, p2, t)
            smoothed.append(pt)
    
    smoothed.append(waypoints[-1])  # Add final point
    return np.array(smoothed)

def get_rrt_path(start, goal, obstacles, gates, time, CTRL_FREQ):
    """
    Computes a smooth path using RRT* through gates with obstacle avoidance.

    Args:
        start (np.ndarray): Starting position (x, y, z).
        goal (np.ndarray): Goal position (x, y, z).
        obstacles (list): Existing obstacle positions with z-coordinate.
        gates (np.ndarray): List of gate positions and yaws.
        time (float): Desired time duration to complete path.
        CTRL_FREQ (float): Control frequency to determine number of waypoints.

    Returns:
        np.ndarray: Smoothed 3D path of shape (N, 3)
    """
    obstacles = [o[:3] for o in obstacles]
    waypoints = augment_waypoints(gates)
    obs = augment_obstacles(gates)
    obstacles = np.vstack((obstacles, obs))

    x_min, x_max = -3.5, 3.5
    y_min, y_max = -3.5, 3.5
    
    planner = RRTStarPlanner(
        obstacles=obstacles,
        x_limits=(x_min, x_max),
        y_limits=(y_min, y_max),
        resolution=0.2,
        obstacle_radius=0.35,
        max_iter=5000,
        step_size=0.05
    )

    waypoints = np.vstack((start[:2], waypoints[:, :2], goal[:2]))
    path = planner.plan_through_waypoints(start[:2], waypoints)
    if path is None:
        print("RRT* could not find a path.")
        return None

    waypoint_count = int(np.round(time * CTRL_FREQ))
    # path = smooth_path_with_cubic_spline(path, num_points=waypoint_count)

    path = np.hstack((path, np.ones((path.shape[0], 1))))  # Add z=1.0
    return path

def get_path(start, goal, obstacles, gates, time, CTRL_FREQ):
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
    waypoint_count = int(np.round(time * CTRL_FREQ))
    path = smooth_path_with_cubic_spline(path, num_points=waypoint_count)
    #path = smooth_path_with_bezier(path, num_points=waypoint_count)
    
    # Add 1.0 as z coordinate to the path
    path = np.hstack((path, np.ones((path.shape[0], 1))))
    
    return path