"""Example utility module.

Please use a file like this one to add extra functions.

"""
import numpy as np
import random
from scipy.interpolate import CubicSpline, splprep, splev



def exampleFunction():
    """Example of user-defined function.

    """
    x = -1
    return x

def generateCircle(radius, start_point, num_points):
    """
    Generates a numpy array of 3D coordinates forming a circle based on a starting point on its edge.

    Args:
        radius (float): Radius of the circle.
        start_point (tuple of float): (x, y, z) coordinates of the starting point on the edge of the circle.
        num_points (int): Number of points forming the circle.

    Returns:
        np.ndarray: A numpy array of shape (num_points, 3) containing the 3D coordinates.
    """
    # Calculate center of the circle assuming the start point is on the edge
    center = (start_point[0] + radius, start_point[1], start_point[2])  # Center is radius units left of start_point

    # Generate the points
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False) + np.pi
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z = np.full_like(x, center[2])  # All points have the same z-coordinate as the center
    return np.column_stack((x, y, z))


STEP_SIZE = 0.2
SEARCH_RADIUS = 0.5
MAX_ITER = 1000
OBSTACLE_RADIUS = 0.085  # 12 cm / 2
GATE_WIDTH = 0.40  # 40 cm gate widt

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

        # Add waypoints to the path
        path = [start]
        for i, wp in enumerate(self.waypoints):
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
            dist = self.distance_point_to_segment(center, p1, p2)
            if dist < (OBSTACLE_RADIUS * 2):
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
    def distance_point_to_segment(point, seg_start, seg_end):
        p = np.array(point)
        a = np.array(seg_start)
        b = np.array(seg_end)
        if np.array_equal(a, b):
            return np.linalg.norm(p - a)
        t = np.dot(p - a, b - a) / np.dot(b - a, b - a)
        t = max(0, min(1, t))
        proj = a + t * (b - a)
        return np.linalg.norm(p - proj)

def augment_obstacles(obstacles, gates):
    """
    Add gate edges as obstacles by representing them as line segments.
    """
    obs = obstacles.copy()
    
    for gate in gates:
        x, y, _, _, _, yaw, _ = gate
        # Calculate the endpoints of the gate edge
        x1 = x - (GATE_WIDTH / 2) - 0.02 * np.cos(yaw)
        y1 = y - (GATE_WIDTH / 2) - 0.02 * np.sin(yaw)
        x2 = x + (GATE_WIDTH / 2) + 0.02 * np.cos(yaw)
        y2 = y + (GATE_WIDTH / 2) + 0.02 * np.sin(yaw)
        
        # Add the line segment as an obstacle (start and end points of the segment)
        obs = np.vstack((obs, [x1, y1, 0, 0, 0, 0]))
        obs = np.vstack((obs, [x2, y2, 0, 0, 0, 0]))
    
    return obs

def augment_waypoints(gates):
    """
    Based on yaw angles, add waypoints to front and back of the gates
    """
    # x, y, z, r, p, y, type
    ways = []
    dist = 0.4
    
    for i, gate in enumerate(gates):
        x, y, z, r, p, yaw, gate_type = gate
        if gate_type == 0:
            # front
            ways.append([x + dist * np.sin(yaw), y + dist * np.cos(yaw), 0, 0, 0, yaw, 0])
            
            # add center waypoint
            ways.append([x, y, 0, 0, 0, yaw, 0])
            # back
            ways.append([x - dist * np.sin(yaw), y - dist * np.cos(yaw), 0, 0, 0, yaw, 0])
        
    return ways

def smooth_cubic_spline(path, obstacles, num_points=300, step=0.05) -> np.ndarray:
    """
    Interpolates through the entire path using cubic splines and checks for obstacle collisions.
    Returns a filtered path with only safe segments.
    """
    def collision_free_segment(p1, p2):
        p1 = np.array(p1)
        p2 = np.array(p2)
        num_checks = max(2, int(np.linalg.norm(p2 - p1) / step))
        for i in range(num_checks + 1):
            interp = p1 + (p2 - p1) * (i / num_checks)
            for obs in obstacles:
                if np.linalg.norm(interp - obs[:2]) < OBSTACLE_RADIUS*2:
                    return False
        return True

    path = np.array(path)
    if len(path) < 2:
        return path

    # Spline interpolation
    x = path[:, 0]
    y = path[:, 1]
    t = np.linspace(0, 1, len(path))
    cs_x = CubicSpline(t, x)
    cs_y = CubicSpline(t, y)
    t_new = np.linspace(0, 1, num_points)

    x_smooth = cs_x(t_new)
    y_smooth = cs_y(t_new)
    smooth_points = np.column_stack((x_smooth, y_smooth))

    # Now filter segments that are obstacle-free
    filtered = [smooth_points[0]]
    for i in range(1, len(smooth_points)):
        if collision_free_segment(filtered[-1], smooth_points[i]):
            filtered.append(smooth_points[i])
        else:
           return None

    return np.array(filtered)

def get_path(start, goal, obstacles, gates):
    obs = augment_obstacles(obstacles, gates)
    
    # conrt obs to only 1st 3 columns
    obs = obs[:, :3]
    waypoints = augment_waypoints(gates)
    
    rrt = RRTStar(x_range=(-3.5, 3.5), y_range=(-3.5, 3.5), obstacles=obs, waypoints=waypoints)
    path = rrt.plan(start[:2], goal[:2])
    
    # Smooth the path using cubic splines
    while True:
        smoothed_path = smooth_cubic_spline(path, obs)
        if smoothed_path is not None:
            break
        else:
            path = rrt.plan(start[:2], goal[:2])
    # append z coordinate as 1
    smoothed_path = np.hstack((smoothed_path, np.ones((smoothed_path.shape[0], 1)) * 1))
    
    
    return smoothed_path
 