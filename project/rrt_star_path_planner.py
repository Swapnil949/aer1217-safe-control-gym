import numpy as np
import random
from shapely.geometry import Point, LineString

class Node:
    def __init__(self, pos):
        self.pos = pos
        self.parent = None
        self.cost = 0.0

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def steer(p1, p2, step_size):
    p1, p2 = np.array(p1), np.array(p2)
    d = np.linalg.norm(p2 - p1)
    if d <= step_size:
        return tuple(p2)
    return tuple(p1 + step_size * (p2 - p1) / d)

def is_collision(p1, p2, obstacles, margin=0.3):
    line = LineString([p1, p2])
    for ox, oy in obstacles:
        if line.distance(Point(ox, oy)) <= margin:
            return True
    return False

def get_nearest(tree, point):
    return min(tree, key=lambda n: distance(n.pos, point))

def get_near(tree, point, radius):
    return [n for n in tree if distance(n.pos, point) < radius]

def extract_path(end_node):
    path = []
    node = end_node
    while node is not None:
        path.append(node.pos)
        node = node.parent
    return path[::-1]

def rrt_star(start, goal, obstacles, bounds, max_iter=500, step_size=0.5, radius=1.0):
    tree = [Node(start)]
    for _ in range(max_iter):
        rnd = (random.uniform(bounds[0], bounds[1]), random.uniform(bounds[2], bounds[3]))
        nearest = get_nearest(tree, rnd)
        new_pos = steer(nearest.pos, rnd, step_size)
        if is_collision(nearest.pos, new_pos, obstacles):
            continue

        new_node = Node(new_pos)
        near_nodes = get_near(tree, new_pos, radius)
        min_cost = nearest.cost + distance(nearest.pos, new_pos)
        best_parent = nearest
        for near in near_nodes:
            cost = near.cost + distance(near.pos, new_pos)
            if cost < min_cost and not is_collision(near.pos, new_pos, obstacles):
                best_parent = near
                min_cost = cost
        new_node.parent = best_parent
        new_node.cost = min_cost
        tree.append(new_node)

        if distance(new_pos, goal) < step_size and not is_collision(new_pos, goal, obstacles):
            goal_node = Node(goal)
            goal_node.parent = new_node
            return extract_path(goal_node)
    return None

def plan_rrt_through_gates(start, gates, goal, obstacles, bounds):
    full_path = []
    all_waypoints = [start] + gates + [goal]

    for i in range(len(all_waypoints) - 1):
        segment_start = all_waypoints[i]
        segment_goal = all_waypoints[i + 1]
        path = rrt_star(segment_start, segment_goal, obstacles, bounds)
        if path is None:
            raise RuntimeError(f"Failed to find path from {segment_start} to {segment_goal}")
        full_path.append(segment_start)
    full_path.append(goal)
    return full_path, all_waypoints

def simplify_path(path, angle_threshold=10):
    if len(path) <= 2:
        return path

    def angle(p1, p2, p3):
        a, b, c = np.array(p1), np.array(p2), np.array(p3)
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    simplified = [path[0]]
    for i in range(1, len(path) - 1):
        if angle(path[i - 1], path[i], path[i + 1]) < (180 - angle_threshold):
            simplified.append(path[i])
    simplified.append(path[-1])
    return simplified

def force_gates(path, gates, min_distance=0.2):
    """
    Ensure all gate centers are present in the path.
    If a gate is missing, insert it at the closest location in the path.
    """
    path = path.copy()
    for gate in gates:
        found = any(np.linalg.norm(np.array(gate) - np.array(p)) < min_distance for p in path)
        if not found:
            dists = [np.linalg.norm(np.array(gate) - np.array(p)) for p in path]
            idx = np.argmin(dists)
            path.insert(idx, gate)
    return path
