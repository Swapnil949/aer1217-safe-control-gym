"""Write your proposed algorithm.
[NOTE]: The idea for the final project is to plan the trajectory based on a sequence of gates 
while considering the uncertainty of the obstacles. The students should show that the proposed 
algorithm is able to safely navigate a quadrotor to complete the task in both simulation and
real-world experiments.

Then run:

    $ python3 final_project.py --overrides ./getting_started.yaml

Tips:
    Search for strings `INSTRUCTIONS` and `REPLACE THIS (START)` in this file.

    Change the code between the 5 blocks starting with
        #########################
        # REPLACE THIS (START) ##
        #########################
    and ending with
        #########################
        # REPLACE THIS (END) ####
        #########################
    with your own code.

    They are in methods:
        1) planning
        2) cmdFirmware

"""
import numpy as np

from collections import deque

try:
    from project_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory
except ImportError:
    # PyTest import.
    from .project_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory

#########################
# REPLACE THIS (START) ##
#########################

# Optionally, create and import modules you wrote.
# Please refrain from importing large or unstable 3rd party packages.
try:
    import example_custom_utils as ecu
except ImportError:
    # PyTest import.
    from . import example_custom_utils as ecu

from example_custom_utils import generate_minimal_gate_path, simplify_path
from pso_path_planner import generate_final_gate_spline_trajectory 
from rrt_star_path_planner import (
    plan_rrt_through_gates,
    simplify_path,
    force_gates
)
from minimum_snap_optimizer import generate_minimum_snap_trajectory
import matplotlib.pyplot as plt


from scipy.interpolate import CubicSpline
#########################
# REPLACE THIS (END) ####
#########################

def plot_debug_path(rrt_path, ref_x, ref_y, gates=None, obstacles=None):
        rrt_path = np.array(rrt_path)

        plt.figure(figsize=(8, 6))

        # RRT waypoints (before smoothing)
        if len(rrt_path) > 0:
            plt.plot(rrt_path[:, 0], rrt_path[:, 1], 'ko--', label='RRT* Waypoints')

        # Smoothed trajectory (final path)
        plt.plot(ref_x, ref_y, 'b-', linewidth=2, label='Minimum Snap Path')

        # Gates
        if gates:
            gx, gy = zip(*gates)
            plt.scatter(gx, gy, c='green', s=80, marker='s', label='Gates')

        # Obstacles
        if obstacles:
            ox, oy = zip(*obstacles)
            plt.scatter(ox, oy, c='red', s=80, marker='x', label='Obstacles')

        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title('Planned Path with Gates and Obstacles')
        plt.grid(True)
        plt.axis("equal")
        plt.legend()
        plt.tight_layout()
        plt.show()

class Controller():
    """Template controller class.

    """

    def __init__(self,
                 initial_obs,
                 initial_info,
                 use_firmware: bool = False,
                 buffer_size: int = 100,
                 verbose: bool = False
                 ):
        """Initialization of the controller.

        INSTRUCTIONS:
            The controller's constructor has access the initial state `initial_obs` and the a priori infromation
            contained in dictionary `initial_info`. Use this method to initialize constants, counters, pre-plan
            trajectories, etc.

        Args:
            initial_obs (ndarray): The initial observation of the quadrotor's state
                [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            initial_info (dict): The a priori information as a dictionary with keys
                'symbolic_model', 'nominal_physical_parameters', 'nominal_gates_pos_and_type', etc.
            use_firmware (bool, optional): Choice between the on-board controll in `pycffirmware`
                or simplified software-only alternative.
            buffer_size (int, optional): Size of the data buffers used in method `learn()`.
            verbose (bool, optional): Turn on and off additional printouts and plots.

        """
        # Save environment and control parameters.
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size

        # Store a priori scenario information.
        # plan the trajectory based on the information of the (1) gates and (2) obstacles. 
        self.NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]

        # Check for pycffirmware.
        if use_firmware:
            self.ctrl = None
        else:
            # Initialize a simple PID Controller for debugging and test.
            # Do NOT use for the IROS 2022 competition. 
            self.ctrl = PIDController()
            # Save additonal environment parameters.
            self.KF = initial_info["quadrotor_kf"]

        # Reset counters and buffers.
        self.reset()
        self.interEpisodeReset()

        # perform trajectory planning
        t_scaled = self.planning(use_firmware, initial_info)

        ## visualization
        # Plot trajectory in each dimension and 3D.
        plot_trajectory(t_scaled, self.waypoints, self.ref_x, self.ref_y, self.ref_z)

        # Draw the trajectory on PyBullet's GUI.
        draw_trajectory(initial_info, self.waypoints, self.ref_x, self.ref_y, self.ref_z)

    

    # def planning(self, use_firmware, initial_info):
    
    #     start = np.array([self.initial_obs[0], self.initial_obs[2], 1.0])
    #     goal = np.array(initial_info["x_reference"])[[0, 2, 4]]
    #     gates = [(g[0], g[1]) for g in self.NOMINAL_GATES]
    #     obstacles = [(o[0], o[1]) for o in self.NOMINAL_OBSTACLES]

        
    #     self.ref_x, self.ref_y, self.ref_z, self.waypoints = generate_minimal_gate_path(
    #         start=start,
    #         goal=goal,
    #         gates=gates,
    #         obstacles=obstacles,
    #         ctrl_freq=self.CTRL_FREQ,
    #         duration=20
    #     )

        
    #     descend_steps = int(2 * self.CTRL_FREQ)
    #     end_idx = len(self.ref_z)
    #     if end_idx > descend_steps:
    #         z_start = self.ref_z[-descend_steps]
    #         self.ref_z[-descend_steps:] = np.linspace(z_start, 0.0, descend_steps)

        
    #     for _ in range(int(3 * self.CTRL_FREQ)):  # 3 seconds of stationary
    #         self.ref_x = np.append(self.ref_x, self.ref_x[-1])
    #         self.ref_y = np.append(self.ref_y, self.ref_y[-1])
    #         self.ref_z = np.append(self.ref_z, self.ref_z[-1])

    #     return np.linspace(0, 1, len(self.ref_x))
    def planning(self, use_firmware, initial_info):
        start = (self.initial_obs[0], self.initial_obs[2])
        goal = tuple(initial_info["x_reference"][[0, 2]])
        gates = [(g[0], g[1]) for g in self.NOMINAL_GATES]
        obstacles = [(o[0], o[1]) for o in self.NOMINAL_OBSTACLES]
        bounds = (-3.5, 3.5, -3.5, 3.5)

        # 1. Generate safe gate-to-gate transitions
        path_2d, _ = plan_rrt_through_gates(start, gates, goal, obstacles, bounds)

        # 2. Clean up path
        path_2d = simplify_path(path_2d, angle_threshold=15)

        # 3. Force gate centers into path (if not already)
        path_2d = force_gates(path_2d, gates)

        # 4. Generate minimum snap trajectory through these critical waypoints
        self.ref_x, self.ref_y, t_scaled = generate_minimum_snap_trajectory(
            path_2d,
            total_time=20,
            ctrl_freq=self.CTRL_FREQ
        )
        self.ref_z = np.full_like(self.ref_x, 1.0)

        # Final reference for controller
        self.waypoints = np.array([(x, y, 1.0) for x, y in path_2d])

        plot_debug_path(path_2d, self.ref_x, self.ref_y, gates=gates, obstacles=obstacles)

        return t_scaled


    def cmdFirmware(self,
                    time,
                    obs,
                    reward=None,
                    done=None,
                    info=None
                    ):
        """Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface.

        INSTRUCTIONS:
            Re-implement this method to return the target position, velocity, acceleration, attitude, and attitude rates to be sent
            from Crazyswarm to the Crazyflie using, e.g., a `cmdFullState` call.

        Args:
            time (float): Episode's elapsed time, in seconds.
            obs (ndarray): The quadrotor's Vicon data [x, 0, y, 0, z, 0, phi, theta, psi, 0, 0, 0].
            reward (float, optional): The reward signal.
            done (bool, optional): Wether the episode has terminated.
            info (dict, optional): Current step information as a dictionary with keys
                'constraint_violation', 'current_target_gate_pos', etc.

        Returns:
            Command: selected type of command (takeOff, cmdFullState, etc., see Enum-like class `Command`).
            List: arguments for the type of command (see comments in class `Command`)

        """
        if self.ctrl is not None:
            raise RuntimeError("[ERROR] Using method 'cmdFirmware' but Controller was created with 'use_firmware' = False.")

        # [INSTRUCTIONS] 
        # self.CTRL_FREQ is 30 (set in the getting_started.yaml file) 
        # control input iteration indicates the number of control inputs sent to the quadrotor
        iteration = int(time*self.CTRL_FREQ)

        #########################
        # REPLACE THIS (START) ##
        #########################

        # print("The info. of the gates ")
        # print(self.NOMINAL_GATES)

        if iteration == 0:
            command_type = Command(2)  # Takeoff
            args = [1.0, 2.0]

        elif 3 * self.CTRL_FREQ <= iteration < 23 * self.CTRL_FREQ:
            step = min(iteration - 3 * self.CTRL_FREQ, len(self.ref_x) - 1)
            pos = [self.ref_x[step], self.ref_y[step], self.ref_z[step]]
            command_type = Command(1)  # cmdFullState
            args = [pos, np.zeros(3), np.zeros(3), 0.0, np.zeros(3)]

        elif iteration == 23 * self.CTRL_FREQ:
            command_type = Command(6)  # NotifySetpointStop
            args = []

        elif iteration == 23 * self.CTRL_FREQ + 1:
            x, y, z = self.ref_x[-1], self.ref_y[-1], 1.5
            command_type = Command(5)  # goTo
            args = [[x, y, z], 0.0, 2.5, False]

        elif iteration == 26 * self.CTRL_FREQ:
            x, y = self.initial_obs[0], self.initial_obs[2]
            command_type = Command(5)  # goTo back
            args = [[x, y, 1.5], 0.0, 4, False]

        elif iteration == 30 * self.CTRL_FREQ:
            command_type = Command(3)  # Land
            args = [0.0, 3.0]

        elif iteration == 33 * self.CTRL_FREQ - 1:
            command_type = Command(4)  # STOP
            args = []

        else:
            command_type = Command(0)  # None
            args = []

        #########################
        # REPLACE THIS (END) ####
        #########################

        return command_type, args

    def cmdSimOnly(self,
                   time,
                   obs,
                   reward=None,
                   done=None,
                   info=None
                   ):
        """PID per-propeller thrusts with a simplified, software-only PID quadrotor controller.

        INSTRUCTIONS:
            You do NOT need to re-implement this method for the project.
            Only re-implement this method when `use_firmware` == False to return the target position and velocity.

        Args:
            time (float): Episode's elapsed time, in seconds.
            obs (ndarray): The quadrotor's state [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            reward (float, optional): The reward signal.
            done (bool, optional): Wether the episode has terminated.
            info (dict, optional): Current step information as a dictionary with keys
                'constraint_violation', 'current_target_gate_pos', etc.

        Returns:
            List: target position (len == 3).
            List: target velocity (len == 3).

        """
        if self.ctrl is None:
            raise RuntimeError("[ERROR] Attempting to use method 'cmdSimOnly' but Controller was created with 'use_firmware' = True.")

        iteration = int(time*self.CTRL_FREQ)

        #########################
        if iteration < len(self.ref_x):
            target_p = np.array([self.ref_x[iteration], self.ref_y[iteration], self.ref_z[iteration]])
        else:
            target_p = np.array([self.ref_x[-1], self.ref_y[-1], self.ref_z[-1]])
        target_v = np.zeros(3)
        #########################

        return target_p, target_v

    def reset(self):
        """Initialize/reset data buffers and counters.

        Called once in __init__().

        """
        # Data buffers.
        self.action_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.obs_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.reward_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.done_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.info_buffer = deque([], maxlen=self.BUFFER_SIZE)

        # Counters.
        self.interstep_counter = 0
        self.interepisode_counter = 0

    # NOTE: this function is not used in the course project. 
    def interEpisodeReset(self):
        """Initialize/reset learning timing variables.

        Called between episodes in `getting_started.py`.

        """
        # Timing stats variables.
        self.interstep_learning_time = 0
        self.interstep_learning_occurrences = 0
        self.interepisode_learning_time = 0

    