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


STEP_SIZE = 0.2
SEARCH_RADIUS = 0.5
MAX_ITER = 1000
SQUARE_OBS_HALF_SIZE = 0.1  # Half of 12cm
GATE_WIDTH = 0.40  # 40 cm gate width

class Node:
    def __init__(self, pos):
        self.pos = pos
        self.parent = None
        self.cost = 0.0

class RRTStar:
    def __init__(self, x_range, y_range, obstacles, waypoints=None):
        self.x_range = x_range
        self.y_range = y_range
        self.obstacles = obstacles
        if waypoints is None:
            waypoints = []
        self.waypoints = [w[:2] for w in waypoints]
        self.nodes = []

    def plan(self, start, goal):
        self.nodes = [Node(start)]
        path = [start]
        for wp in self.waypoints:
            path.extend(self.plan_between(path[-1], wp))
        path.extend(self.plan_between(self.waypoints[-1], goal))
        return path

    def plan_between(self, start, goal):
        temp_path = []
        self.nodes = [Node(start)]
        for _ in range(MAX_ITER):
            rand_point = self.sample()
            nearest_node = self.nearest(rand_point)
            new_node = self.steer(nearest_node, rand_point)

            if not self.collision_check(nearest_node.pos, new_node.pos):
                neighbors = self.find_nearby_nodes(new_node)
                min_cost_node = self.choose_best_parent(neighbors, nearest_node, new_node)
                if min_cost_node:
                    new_node.parent = min_cost_node
                    new_node.cost = min_cost_node.cost + np.linalg.norm(np.array(new_node.pos) - np.array(min_cost_node.pos))
                self.nodes.append(new_node)
                self.rewire(new_node, neighbors)

                if np.linalg.norm(np.array(new_node.pos) - np.array(goal)) < STEP_SIZE:
                    final_node = Node(goal)
                    final_node.parent = new_node
                    final_node.cost = new_node.cost + np.linalg.norm(np.array(goal) - np.array(new_node.pos))
                    self.nodes.append(final_node)
                    temp_path = self.extract_path(final_node)
                    break
        return temp_path

    def sample(self):
        return [random.uniform(*self.x_range), random.uniform(*self.y_range)]

    def nearest(self, point):
        return min(self.nodes, key=lambda node: np.linalg.norm(np.array(node.pos) - np.array(point)))

    def steer(self, from_node, to_point):
        vec = np.array(to_point) - np.array(from_node.pos)
        dist = np.linalg.norm(vec)
        direction = vec / dist if dist != 0 else vec
        new_pos = from_node.pos + STEP_SIZE * direction
        new_node = Node(new_pos.tolist())
        new_node.parent = from_node
        new_node.cost = from_node.cost + STEP_SIZE
        return new_node

    def collision_check(self, p1, p2):
        for obs in self.obstacles:
            center = obs[:2]
            min_x, max_x = center[0] - SQUARE_OBS_HALF_SIZE, center[0] + SQUARE_OBS_HALF_SIZE
            min_y, max_y = center[1] - SQUARE_OBS_HALF_SIZE, center[1] + SQUARE_OBS_HALF_SIZE
            if self.segment_intersects_box(p1, p2, min_x, max_x, min_y, max_y):
                return True
        return False

    def find_nearby_nodes(self, new_node):
        radius = SEARCH_RADIUS
        return [node for node in self.nodes if np.linalg.norm(np.array(node.pos) - np.array(new_node.pos)) < radius]

    def choose_best_parent(self, neighbors, nearest_node, new_node):
        min_cost = nearest_node.cost + np.linalg.norm(np.array(new_node.pos) - np.array(nearest_node.pos))
        best_node = nearest_node
        for node in neighbors:
            cost = node.cost + np.linalg.norm(np.array(new_node.pos) - np.array(node.pos))
            if cost < min_cost and not self.collision_check(node.pos, new_node.pos):
                min_cost = cost
                best_node = node
        return best_node

    def rewire(self, new_node, neighbors):
        for node in neighbors:
            cost = new_node.cost + np.linalg.norm(np.array(node.pos) - np.array(new_node.pos))
            if cost < node.cost and not self.collision_check(new_node.pos, node.pos):
                node.parent = new_node
                node.cost = cost

    def extract_path(self, node):
        path = []
        while node:
            path.append(node.pos)
            node = node.parent
        return path[::-1]

    @staticmethod
    def segment_intersects_box(p1, p2, min_x, max_x, min_y, max_y):
        x1, y1 = p1
        x2, y2 = p2

        def clip(p, q, u1, u2):
            if p == 0:
                return q >= 0, u1, u2
            u = q / p
            if p < 0:
                if u > u2: return False, u1, u2
                if u > u1: u1 = u
            else:
                if u < u1: return False, u1, u2
                if u < u2: u2 = u
            return True, u1, u2

        dx = x2 - x1
        dy = y2 - y1
        u1, u2 = 0.0, 1.0

        for p, q in [(-dx, x1 - min_x), (dx, max_x - x1),
                     (-dy, y1 - min_y), (dy, max_y - y1)]:
            valid, u1, u2 = clip(p, q, u1, u2)
            if not valid:
                return False
        return True


def augment_obstacles(obstacles, gates):
    obs = obstacles.copy()
    for gate in gates:
        x, y, _, _, _, yaw, _ = gate
        x1 = x - (GATE_WIDTH / 2) - 0.02 * np.cos(yaw)
        y1 = y - (GATE_WIDTH / 2) - 0.02 * np.sin(yaw)
        x2 = x + (GATE_WIDTH / 2) + 0.02 * np.cos(yaw)
        y2 = y + (GATE_WIDTH / 2) + 0.02 * np.sin(yaw)
        obs = np.vstack((obs, [x1, y1, 0, 0, 0, 0]))
        obs = np.vstack((obs, [x2, y2, 0, 0, 0, 0]))
    return obs

def augment_waypoints(gates):
    ways = []
    dist = 0.3
    for gate in gates:
        x, y, z, r, p, yaw, gate_type = gate
        if gate_type == 0:
            ways.append([x + dist * np.sin(yaw), y + dist * np.cos(yaw), 0, 0, 0, yaw, 0])
            ways.append([x, y, 0, 0, 0, yaw, 0])
            ways.append([x - dist * np.sin(yaw), y - dist * np.cos(yaw), 0, 0, 0, yaw, 0])
    return ways

def smooth_cubic_spline(path, obstacles, num_points=500, step=0.01) -> np.ndarray:
    def collision_free_segment(p1, p2):
        p1 = np.array(p1)
        p2 = np.array(p2)
        num_checks = max(2, int(np.linalg.norm(p2 - p1) / step))
        for i in range(num_checks + 1):
            interp = p1 + (p2 - p1) * (i / num_checks)
            for obs in obstacles:
                cx, cy = obs[:2]
                if (cx - SQUARE_OBS_HALF_SIZE <= interp[0] <= cx + SQUARE_OBS_HALF_SIZE and
                    cy - SQUARE_OBS_HALF_SIZE <= interp[1] <= cy + SQUARE_OBS_HALF_SIZE):
                    return False
        return True

    path = np.array(path)
    if len(path) < 2:
        return path

    x = path[:, 0]
    y = path[:, 1]
    t = np.linspace(0, 1, len(path))
    cs_x = CubicSpline(t, x)
    cs_y = CubicSpline(t, y)
    t_new = np.linspace(0, 1, num_points)
    x_smooth = cs_x(t_new)
    y_smooth = cs_y(t_new)
    smooth_points = np.column_stack((x_smooth, y_smooth))

    filtered = [smooth_points[0]]
    for i in range(1, len(smooth_points)):
        if collision_free_segment(filtered[-1], smooth_points[i]):
            filtered.append(smooth_points[i])
        else:
            return None
    return np.array(filtered)

def get_path(start, goal, obstacles, gates):
    obs = augment_obstacles(obstacles, gates)
    obs = obs[:, :3]
    waypoints = augment_waypoints(gates)
    rrt = RRTStar(x_range=(-3.5, 3.5), y_range=(-3.5, 3.5), obstacles=obs, waypoints=waypoints)
    path = rrt.plan(start[:2], goal[:2])
    while True:
        smoothed_path = smooth_cubic_spline(path, obs)
        if smoothed_path is not None:
            break
        else:
            print("Smoothing failed, retrying...")
            path = rrt.plan(start[:2], goal[:2])
    smoothed_path = np.hstack((smoothed_path, np.ones((smoothed_path.shape[0], 1)) * 1))
    return smoothed_path