import numpy as np
import cvxpy as cp
import math

def generate_minimum_snap_trajectory(waypoints, total_time=20, ctrl_freq=30):
    """
    Generate a 7th-order minimum snap trajectory using polynomial optimization.
    Interpolates through all given 2D waypoints.
    """

    waypoints = np.array(waypoints)
    N = len(waypoints) - 1  # Number of segments
    d_order = 4             # Snap = 4th derivative
    n_coef = 8              # 7th-order polynomial

    # Time allocation (uniform for now)
    times = np.linspace(0, total_time, N + 1)
    segment_durations = np.diff(times)

    # Decision variables: coeffs for each segment
    coeffs_x = cp.Variable((N, n_coef))
    coeffs_y = cp.Variable((N, n_coef))

    def poly_derivative_vec(t, order):
        return np.array([
            math.factorial(i) / math.factorial(i - order) * t**(i - order) if i >= order else 0
            for i in range(n_coef)
        ])

    def snap_cost_matrix(T):
        Q = np.zeros((n_coef, n_coef))
        for i in range(4, n_coef):
            for j in range(4, n_coef):
                Q[i, j] = (
                    math.factorial(i) * math.factorial(j) /
                    (math.factorial(i - 4) * math.factorial(j - 4) * (i + j - 7))
                ) * T**(i + j - 7)
        return Q

    # === Objective: minimize total snap ===
    cost = 0
    for i in range(N):
        Q = snap_cost_matrix(segment_durations[i])
        cost += cp.sum_squares(Q @ coeffs_x[i]) + cp.sum_squares(Q @ coeffs_y[i])

    constraints = []

    # === Waypoint position constraints (start + end) ===
    for i in range(N):
        t0 = 0
        tf = segment_durations[i]
        A0 = np.array([t0**j for j in range(n_coef)])
        Af = np.array([tf**j for j in range(n_coef)])

        constraints += [
            coeffs_x[i] @ A0 == waypoints[i, 0],
            coeffs_x[i] @ Af == waypoints[i+1, 0],
            coeffs_y[i] @ A0 == waypoints[i, 1],
            coeffs_y[i] @ Af == waypoints[i+1, 1],
        ]

    # === Continuity at internal waypoints (vel, acc, jerk) ===
    for i in range(N - 1):
        for d in range(1, d_order):  # 1st to 3rd derivative
            Df = poly_derivative_vec(segment_durations[i], d)
            D0 = poly_derivative_vec(0, d)
            constraints += [
                coeffs_x[i] @ Df == coeffs_x[i+1] @ D0,
                coeffs_y[i] @ Df == coeffs_y[i+1] @ D0,
            ]

    # === Solve the convex optimization ===
    try:
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP)
    except cp.SolverError:
        print("[INFO] OSQP failed, retrying with SCS...")
        prob.solve(solver=cp.SCS)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError("Minimum snap trajectory optimization failed.")

    # === Evaluate trajectory over time ===
    dense_time = np.linspace(0, total_time, int(total_time * ctrl_freq))
    x_vals, y_vals = [], []
    t_cursor = 0

    for i in range(N):
        T = segment_durations[i]
        t_vals = dense_time[(dense_time >= t_cursor) & (dense_time <= t_cursor + T)] - t_cursor
        powers = np.array([t_vals**j for j in range(n_coef)])
        x_seg = coeffs_x.value[i] @ powers
        y_seg = coeffs_y.value[i] @ powers
        x_vals.extend(x_seg)
        y_vals.extend(y_seg)
        t_cursor += T

    return np.array(x_vals), np.array(y_vals), dense_time / total_time
