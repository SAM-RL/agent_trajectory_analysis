import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
import time
import scipy.io as io
import random
import os

class CustomDifusionFieldEnvironment:
    """
    This environment is similar to the regular field, except the state vector contains the gradient
    and concentration values, but does not contain location, and has 9 actions
    """

    def __init__(
            self,
            learning_experiment_name,
            field_size=[100, 100],
            max_num_steps=500,
            view_scope_half_side=5,
            num_sources=1,
            initial_field_path=None,
            params={}):
        
        # Set advection diffusion parameters
        self.dx = params.get("dx", 0.8)
        self.dy = params.get("dy", 0.8)
        self.vx = params.get("vx", 0.7)  # 3: -0.7, 4: 0.7, 2: -0.6
        self.vy = params.get("vy", -0.4)  # 3: -0.3, 4: -0.4, 2: 0.8
        self.dt = params.get("dt", 0.1)
        self.k = params.get("k", 1.0)

        # Set field grid params/variables
        self.field_size =  params.get("field_size", [100, 100])
        self.max_num_steps = params.get("max_steps", 400)
        self.view_scope_half_side = params.get("scope_half_size", 5)
        self.field_area = self.field_size[0] * self.field_size[1]

        # Environment's field params
        self.num_sources = params.get("n_source", 1)
        self.field_path = params.get("field_path", None)
        assert(self.field_path != None)

        self.env_curr_field = None
        
        with open(self.field_path, 'rb') as f:
            initial_field = np.load(f)
            self.env_curr_field = initial_field

        self.template_peak_conc = np.max(self.env_curr_field)
        self.env_prev_field = np.zeros(self.field_size)

        # Agent Field related params/variables
        self.num_steps = 0
        self.agent_start_position = None
        self.agent_position = None
        self.agent_curr_field = np.zeros(self.field_size)
        self.agent_field_visited = np.zeros(self.field_size)
        self.agent_trajectory = []
        self.actions_list = []
        self.curr_view_scope = np.zeros(
            [2 * self.view_scope_half_side + 1, 2 * self.view_scope_half_side + 1])
        self.agent_gradients = [0.0, 0.0]

        # Statistics variables
        self.agent_coverage = []
        self.rewards = []
        self.mapping_errors = []
        self.concentrations = []
        self.gradients_0 = []
        self.gradients_1 = []

        # Environment Observation space. This includes x, y, conc, grad_x, grad_y as state
        low = np.array([0.0, -100.0, -100.0])
        high = np.array([25.0, 100.0, 100.0])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Agent Action related params/variables
        self.action_space_map = {}
        self.inversed_action_space_map = {}
        self.actions = ["left", "right", "up", "down", "stay",
                        "up-left", "up-right", "down-left", "down-right"]
        self.action_space = spaces.Discrete(9)
        for action_id, action in enumerate(self.actions):
            self.action_space_map[action_id] = action
            self.inversed_action_space_map[action] = action_id

        # Misc. params
        self.learning_experiment_name = learning_experiment_name

        # Initial mapping error
        self.init_mapping_error = np.sum(self.env_curr_field)
    
    def update_env_field(self):
        updated_u = self.env_curr_field.copy()
        u_k = self.env_curr_field.copy()

        dx = self.dx
        dy = self.dy
        dt = self.dt
        vx = self.vx
        vy = self.vy
        k = self.k

        # fmt: off
        u_k[0,:] = u_k[1,:]
        u_k[:,0] = u_k[:,1]
        u_k[-1,:] = u_k[-2,:]
        u_k[:,-1] = u_k[:,-2]

        u_k[0,0] = u_k[1,1]
        u_k[-1,-1] = u_k[-2,-2]
        u_k[0,-1] = u_k[1,-2]
        u_k[-1,0] = u_k[-2,1]

        for i in range(1, self.field_size[0] - 1):
            for j in range(1, self.field_size[1] - 1):
                updated_u[j, i] = u_k[j, i] + k * (dt / dx ** 2) * \
                    ((u_k[j + 1, i] + u_k[j - 1, i] +
                      u_k[j, i + 1] + u_k[j, i - 1] - 4 * u_k[j, i])) + \
                    vx * (dt / dx) * ((u_k[j + 1, i] - u_k[j, i])) + vy * (dt / dy) * \
                    (u_k[j, i + 1] - u_k[j, i])

        self.env_curr_field = updated_u

    def update_field(self, field):
        updated_u = field.copy()
        u_k = field.copy()

        dx = self.dx
        dy = self.dy
        dt = self.dt
        vx = self.vx
        vy = self.vy
        k = self.k
        
        # fmt: off
        for i in range(1, self.field_size[0] - 1):
            for j in range(1, self.field_size[1] - 1):
                updated_u[j, i] = u_k[j, i] + k * (dt / dx ** 2) * \
                    ((u_k[j + 1, i] + u_k[j - 1, i] +
                      u_k[j, i + 1] + u_k[j, i - 1] - 4 * u_k[j, i])) + \
                    vx * (dt / dx) * ((u_k[j + 1, i] - u_k[j, i])) + vy * (dt / dy) * \
                    (u_k[j, i + 1] - u_k[j, i])
        # fmt: on                                                                                                                                                                                          i])

        # self.env_prev_field = self.env_curr_field
        return updated_u

    def calculate_gradients(self, r):
        dz_dx = (self.agent_curr_field[r[0] + 1, r[1]] -
                 self.agent_curr_field[r[0] - 1, r[1]]) / (2 * self.dx)
        dz_dy = (self.agent_curr_field[r[0], r[1] + 1] -
                 self.agent_curr_field[r[0], r[1] - 1]) / (2 * self.dy)

        return np.array([dz_dx, dz_dy])

    def step(self, action_id):
        # Ensure action is a valid action and exists in Agent's action space
        assert self.action_space.contains(
            action_id), "Action %r (%s) is invalid!" % (action_id, type(action_id))

        action = self.action_space_map[action_id]
        assert action in self.actions, "%s (%s) invalid" % (
            action, type(action))

        self.actions_list.append(action_id)

        # Get the next state
        (hit_wall, next_position) = self.get_next_position(action)

        # Update field state
        self.update_env_field()

        # Update agent's view of the field
        self.update_agent_field_and_coverage(next_position)

        # Update Mapping error
        curr_mapping_error = self.calculate_mapping_error()
        self.mapping_errors.append(curr_mapping_error)

        # Get gradients
        self.agent_gradients = self.calculate_gradients(self.agent_position)

        # Update number of steps
        self.num_steps += 1

        # Check for termination criteria
        done = False
        if (self.num_steps >= self.max_num_steps):
            done = True

        # Get any observations
        observations = {"location": next_position}

        # Update agent variables
        self.agent_position = next_position
        self.agent_trajectory.append(self.agent_position)

        # Get concentration
        concentration = self.env_curr_field[self.agent_position[0],
                                            self.agent_position[1]]

        # Record field values
        self.concentrations.append(concentration)
        self.gradients_0.append(self.agent_gradients[0])
        self.gradients_1.append(self.agent_gradients[1])

        return ([concentration, self.agent_gradients[0], self.agent_gradients[1]], done, observations)

    def reset(self, start_pos=[80, 90]):
        # Reset agent related params
        self.num_steps = 0
        self.agent_position = start_pos
        self.agent_curr_field = np.zeros(self.field_size)
        self.agent_field_visited = np.zeros(self.field_size)
        self.agent_trajectory = []
        self.curr_view_scope = np.zeros(
            [2 * self.view_scope_half_side + 1, 2 * self.view_scope_half_side + 1])
        self.agent_gradients = [0.0, 0.0]
        
        if (self.field_path is None):
            self.env_curr_field = self.create_field()
        else:
            with open(self.field_path, 'rb') as f:
                initial_field = np.load(f)
                self.env_curr_field = initial_field

        # Reset stats
        self.agent_coverage = []
        self.mapping_errors = []
        self.concentrations = []
        self.gradients_0 = []
        self.gradients_1 = []

        # Return the first state
        # Get concentration
        concentration = self.env_curr_field[self.agent_position[0],
                                            self.agent_position[1]]

        # Get gradients
        self.agent_gradients = self.calculate_gradients(self.agent_position)

        # Record field values
        self.concentrations.append(concentration)
        self.gradients_0.append(self.agent_gradients[0])
        self.gradients_1.append(self.agent_gradients[1])

        return [concentration, self.agent_gradients[0], self.agent_gradients[1]]
    
    def get_next_action_probs(self, next_state=None):
        next_state = self.agent_position if next_state is None else next_state
        action_probs = np.ones(len(self.actions))
        invalid_actions = set()
        if next_state[1] < (0 + self.view_scope_half_side) + 1:
            invalid_actions.update({"left", "up-left", "down-left"})
        if next_state[1] >= (self.field_size[1] - self.view_scope_half_side) - 1:
            invalid_actions.update({"right", "up-right", "down-right"})
        if next_state[0] < (0 + self.view_scope_half_side) + 1:
            invalid_actions.update({"up", "up-left", "up-right"})
        if next_state[0] >= (self.field_size[0] - self.view_scope_half_side) - 1:
            invalid_actions.update({"down", "down-left", "down-right"})
        for action in invalid_actions:
            action_probs[self.inversed_action_space_map[action]] = 0
        return action_probs        

    def get_next_position(self, action):
        # Create a deepcopy of current state
        next_state = copy.deepcopy(self.agent_position)

        # Only update the next_state if the action changes the position, else stay
        if action == "left":
            next_state[1] = next_state[1] - 1
        elif action == "right":
            next_state[1] = next_state[1] + 1
        elif action == "up":
            next_state[0] = next_state[0] - 1
        elif action == "down":
            next_state[0] = next_state[0] + 1
        elif action == "stay":
            pass
        elif action == "up-left":
            next_state[0] = next_state[0] - 1
            next_state[1] = next_state[1] - 1
        elif action == "up-right":
            next_state[0] = next_state[0] - 1
            next_state[1] = next_state[1] + 1
        elif action == "down-left":
            next_state[0] = next_state[0] + 1
            next_state[1] = next_state[1] - 1
        elif action == "down-right":
            next_state[0] = next_state[0] + 1
            next_state[1] = next_state[1] + 1

        # Check for collisions
        hit_wall = False
        if ((next_state[0] < (0 + self.view_scope_half_side) or
             next_state[0] >= (self.field_size[0] - self.view_scope_half_side)) or
            ((next_state[1] < (0 + self.view_scope_half_side) or
              next_state[1] >= (self.field_size[1] - self.view_scope_half_side)))):
            # If the view scope is out of the field, hit_wall is set to True
            hit_wall = True

        return (hit_wall, next_state)

    def normalize(self, field):
        max_val = field.max()
        min_val = field.min()
        field_normalized = (field - min_val) / (max_val - min_val)
        return field_normalized

    def calculate_mapping_error(self):
        return np.sum(np.abs(self.agent_curr_field - self.env_curr_field))
    
   
    def update_agent_field_and_coverage(self, next_state):
        vs_min_row = next_state[0] - self.view_scope_half_side
        vs_max_row = next_state[0] + self.view_scope_half_side + 1

        vs_min_col = next_state[1] - self.view_scope_half_side
        vs_max_col = next_state[1] + self.view_scope_half_side + 1

        # Count prev_visited
        prev_visited = np.count_nonzero(self.agent_field_visited)

        self.agent_curr_field[vs_min_row:vs_max_row, vs_min_col:vs_max_col] = \
            self.env_curr_field[vs_min_row:vs_max_row, vs_min_col:vs_max_col]

        self.agent_field_visited[vs_min_row:vs_max_row,
                                 vs_min_col:vs_max_col] = 1

        self.curr_view_scope = self.agent_curr_field[vs_min_row:vs_max_row,
                                                     vs_min_col:vs_max_col]

        self.agent_curr_field = self.update_field(self.agent_curr_field)

        # Count curr_visited
        curr_visited = np.count_nonzero(self.agent_field_visited)
        frac_coverage_improvement = float(
            curr_visited) - float(prev_visited) / float(self.field_area)

        # Record coverage percentage
        self.agent_coverage.append(
            (float(curr_visited) * 100.0) / float(self.field_area))
        return frac_coverage_improvement
    
    def view_testing_episode_state(self, episode_num, timestep):

        cmap_color = "Blues"

        fig_learning, fig_learning_axes = plt.subplots(1, 2, figsize=(20, 15))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        fig_learning_axes[0].set_title("Environment Field End State")
        fig_learning_axes[0].set_aspect("equal")

        fig_learning_axes[1].set_title("Mapping Error")
        fig_learning_axes[1].set_xlim([0, self.max_num_steps])

        # Plot 1: Environment End state
        fig_learning_axes[0].imshow(
            self.env_curr_field.T, cmap=cmap_color)

        traj_r = [position[1] for position in self.agent_trajectory]
        traj_c = [position[0] for position in self.agent_trajectory]
        fig_learning_axes[0].plot(traj_r, traj_c, '.', color='black')

        # print(self.agent_trajectory)
        fig_learning_axes[0].plot(
            self.agent_trajectory[0][0], self.agent_trajectory[0][1], '*', color='red')
        
        view_scope_box = patches.Rectangle(
            (self.agent_trajectory[-1][1] - self.view_scope_half_side,
             self.agent_trajectory[-1][0] - self.view_scope_half_side), self.view_scope_half_side*2+1, self.view_scope_half_side*2+1,
            linewidth=2, edgecolor='r', facecolor='none')
        fig_learning_axes[0].add_patch(view_scope_box)

        # Plot 2: Mapping Error
        fig_learning_axes[1].plot(self.mapping_errors, '.-')

        # Add Episode number to top of image
        fig_learning.suptitle(
            "Test Episode: " + str(episode_num) + ", at timestep: " + str(timestep))

        plt.show()

        plt.close()

    def render(self, mode="human"):
        pass
