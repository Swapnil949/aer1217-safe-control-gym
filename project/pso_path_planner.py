import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import KDTree
import heapq

# A* algorithm (grid-based)
def a_star(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (1,1), (-1,1), (1,-1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor] == 1:
                    continue
                tentative_g = g_score[current] + heuristic(current, neighbor)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

    return []  # No path found

# Grid builder with obstacles and margin
def build_grid(x_bounds, y_bounds, resolution=0.1, obstacles=[], margin=0.3):
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    grid_x = int((x_max - x_min) / resolution)
    grid_y = int((y_max - y_min) / resolution)
    grid = np.zeros((grid_x, grid_y))

    for obs in obstacles:
        ox, oy = obs[0], obs[1]
        for i in range(grid_x):
            for j in range(grid_y):
                x = x_min + i * resolution
                y = y_min + j * resolution
                if np.linalg.norm([ox - x, oy - y]) < margin:
                    grid[i, j] = 1
    return grid, x_min, y_min, resolution

# Final trajectory generator using enforced gate centers + spline

def generate_final_gate_spline_trajectory(start, goal, gates, obstacles, ctrl_freq=30, duration=20):
    z_height = 1.0
    waypoints = [start[:2]] + gates + [goal[:2]]

    # Optional: enforce obstacle clearance (remove gates near obstacles)
    tree = KDTree(obstacles)
    safe_waypoints = []
    margin = 0.5

    for wp in waypoints:
        d, _ = tree.query(wp, k=1)
        if d >= margin:
            safe_waypoints.append(wp)
        else:
            print(f"[WARNING] Skipping waypoint {wp} too close to obstacle")

    if safe_waypoints[0] != list(start[:2]):
        safe_waypoints.insert(0, list(start[:2]))
    if safe_waypoints[-1] != list(goal[:2]):
        safe_waypoints.append(list(goal[:2]))

    full_path_3d = np.column_stack([safe_waypoints, np.full(len(safe_waypoints), z_height)])

    t = np.linspace(0, 1, full_path_3d.shape[0])
    cs_x = CubicSpline(t, full_path_3d[:, 0])
    cs_y = CubicSpline(t, full_path_3d[:, 1])
    cs_z = CubicSpline(t, full_path_3d[:, 2])

    t_scaled = np.linspace(0, 1, int(duration * ctrl_freq))
    ref_x = cs_x(t_scaled)
    ref_y = cs_y(t_scaled)
    ref_z = cs_z(t_scaled)

    return ref_x, ref_y, ref_z, full_path_3d