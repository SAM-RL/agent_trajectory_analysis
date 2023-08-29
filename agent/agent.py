from .algorithms import SourceDetector, DestinationChooser, Controller
from .dqn import DQN
from .astar import astar
import torch
import numpy as np

# -------------------------------
# UTILITIES
# -------------------------------
ACTION_MAP = {
    (0,-1): 0,  # left
    (0, 1): 1,  # right
    (-1,0): 2,  # up
    (1, 0): 3,  # down
    (0, 0): 4,  # stay
    (-1,-1): 5, # up-left
    (-1, 1): 6, # up-right
    (1, -1): 7, # down-left
    (1, 1): 8, # down-right
}

def generate_agent_trajectory(start=(0,0), end=(0,0), field_size=100):
    path = astar(np.zeros((field_size,field_size)), start, end)
    actions = []
    if len(path) > 0:
        for i in range(len(path)-1):
            offset = (path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])
            actions.append(ACTION_MAP[offset])
    return actions

def bounded_vec(raw_vec, field_size=100, padding=0):
    return np.array([max(min(raw_vec[0], field_size-padding-2), padding), max(min(raw_vec[1], field_size-padding-2), padding)])

# -------------------------------
# AGENT BASE CLASS
# -------------------------------
class BaseAgent():
    def __init__(self, params):
        pass
    def get_next_action(self, state, params):
        pass


# -------------------------------
# SOURCE DETECTOR CONFIG
# -------------------------------
SOURCE_DETECTOR_CONFIG = {
    'CONCENTRATION_THRESH': 13,
    'BUFFER_SIZE': 30,
    'POSITION_BUFFER_SIZE': 50,
    'GRAD_THRESHOLD': 0.3,
}

# -------------------------------
# DQN AGENT
# -------------------------------
DQN_DEFAULT_PARAMS = {
    'MODEL_PATH': './agent/model.dat',
    'CONCENTRATION_THRESH': 6,
    'OBSERVATION_SHAPE': (3,),
    'ACTION_SHAPE': 9,
    'DEVICE': 'cpu',
    'RL_MODE_WARMUP_STEPS': 10
}
class DQNAgent(BaseAgent):
    def __init__(self, params=DQN_DEFAULT_PARAMS):
        # load parameters
        self.model_path = params['MODEL_PATH']
        self.concentration_thresh = params['CONCENTRATION_THRESH']
        self.env_observation_space = params['OBSERVATION_SHAPE']
        self.env_action_space = params['ACTION_SHAPE']
        self.device_name = params['DEVICE']
        # initialize source detector
        self.source_detector = SourceDetector(
            conc_threshold=SOURCE_DETECTOR_CONFIG['CONCENTRATION_THRESH'], buffer_size=SOURCE_DETECTOR_CONFIG['BUFFER_SIZE'],
            position_buffer_size=SOURCE_DETECTOR_CONFIG['POSITION_BUFFER_SIZE'], grad_threshold=SOURCE_DETECTOR_CONFIG['GRAD_THRESHOLD']
        )
        # initialize controllers
        self.total_warmup_steps = params['RL_MODE_WARMUP_STEPS']
        self.warmup_step = 0
        self.destination_chooser = DestinationChooser(30)
        self.actions = []
        self.source_found = False
        self.handling_source_found = False
        self.warmup_steps = 0
        self.detected_sources = []
        # load DQN model
        self.device = torch.device(self.device_name) 
        self.net = DQN(self.env_observation_space, 512, self.env_action_space).to(self.device)
        state = torch.load(self.model_path, map_location=lambda stg, _: stg)
        self.net.load_state_dict(state)

    def get_next_action(self, state, params):
        position, field_visited, field_size, action_probs, n_srcs = params['position'], params['field_visited'], params['field_size'][0], params['action_probs'], params['n_srcs']
        action = None
        while action == None:
            if not self.source_found:     
                self.source_found = self.source_detector.is_source_detected(state, position)
                if self.source_found:
                    self.detected_sources.append(position)
                    continue
                else:
                    # print('--- exploring ---')
                    rl_mode = (state[0] >= self.concentration_thresh) or self.warmup_step
                    # print(f'--- warmup:{self.warmup_step} ---') 
                    if rl_mode:
                        # print('-------> rl_mode ---')
                        if (self.warmup_step == 0):
                            self.warmup_step = self.total_warmup_steps
                        else:
                            self.warmup_step = self.warmup_step - 1
                        self.actions = []
                        state_v = torch.tensor(np.array([state], copy=False)).to(self.device)
                        q_vals = self.net(state_v).data.cpu().numpy()[0]
                        action = np.argmax(q_vals * action_probs)
                    else:
                        # print('-------> controler mode ---')
                        self.warmup_step = 0
                        if len(self.actions) == 0:
                            destination, path_to_destination = self.destination_chooser.find_destination(position, field_visited, self.detected_sources)
                            self.actions = generate_agent_trajectory(start=tuple(position), \
                                                                     end=tuple(bounded_vec(destination)), field_size=field_size)
                        action = self.actions.pop(0)
            else:
                self.warmup_step = 0
                if len(self.detected_sources) >= n_srcs:
                    return 4
                if not self.handling_source_found:
                    # print('--- handling source found ---')
                    destination, path_to_destination = self.destination_chooser.find_destination(position, field_visited, self.detected_sources)
                    self.actions = generate_agent_trajectory(start=tuple(position), \
                                                                 end=tuple(bounded_vec(destination)), field_size=field_size)
                    self.handling_source_found = True
                    action = self.actions.pop(0)
                elif len(self.actions)  == 0:
                    # print('--- reach destination, reset detector ---')
                    self.source_detector.reset()
                    self.source_found = self.handling_source_found = False
                else: 
                    # print('--- get next action from traj ---')
                    action = self.actions.pop(0)                
        return action
        
# -------------------------------
# GRADIENT AGENT
# -------------------------------
GRADIENT_DEFAULT_PARAMS = {
    'CONCENTRATION_THRESH': 6,
    'PARAM_K': 2.0,
}
class GradientAgent(BaseAgent):
    def __init__(self, params=GRADIENT_DEFAULT_PARAMS):
        # load parameters
        self.concentration_thresh = params['CONCENTRATION_THRESH']
        self.k = params['PARAM_K']
        # initialize controllers
        self.source_detector = SourceDetector(
            conc_threshold=SOURCE_DETECTOR_CONFIG['CONCENTRATION_THRESH'], buffer_size=SOURCE_DETECTOR_CONFIG['BUFFER_SIZE'],
            position_buffer_size=SOURCE_DETECTOR_CONFIG['POSITION_BUFFER_SIZE'], grad_threshold=SOURCE_DETECTOR_CONFIG['GRAD_THRESHOLD']
        )
        self.destination_chooser = DestinationChooser(30)
        self.source_found = False
        self.handling_source_found = False
        self.detected_sources = []
        # self.destination_chooser = DestinationChooser(20)
        self.controller = None
        # initialize action_queue
        self.gradient_actions = []
        self.explore_actions = []

    def get_next_action(self, state, params):
        position, field_visited, field_size, n_srcs = np.array(params['position']), params['field_visited'], params['field_size'][0], params['n_srcs']
        action = None
        while action == None:
            if not self.source_found:     
                self.source_found = self.source_detector.is_source_detected(state, position)
                if self.source_found:
                    self.detected_sources.append(position)
                    continue
                else:
                    gradient_mode = state[0] >= self.concentration_thresh
                    if gradient_mode:
                        self.explore_actions = []
                        if len(self.gradient_actions) == 0:
                            grad_vec = np.array([state[1], state[2]])
                            grad_vec_k = self.k * grad_vec / np.linalg.norm(grad_vec)
                            target_position = bounded_vec(position + grad_vec_k, field_size=field_size).round()
                            # print(f"({tuple(position)},{tuple(target_position)})")
                            self.gradient_actions = generate_agent_trajectory(start=tuple(position), \
                                                                              end=tuple(target_position), field_size=field_size)
                        action = self.gradient_actions.pop(0)            
                    else:
                        self.gradient_actions = []
                        if len(self.explore_actions) == 0:
                            destination, path_to_destination = self.destination_chooser.find_destination(position, field_visited)
                            self.explore_actions = generate_agent_trajectory(start=tuple(position), \
                                                         end=tuple(bounded_vec(destination)), field_size=field_size)
                        action = self.explore_actions.pop(0)
                    return action
            else:
                if len(self.detected_sources) >= n_srcs:
                    return 4
                if not self.handling_source_found:
                    # print('--- handling source found ---')
                    destination, path_to_destination = self.destination_chooser.find_destination(position, field_visited)
                    self.actions = generate_agent_trajectory(start=tuple(position), \
                                                                 end=tuple(bounded_vec(destination)), field_size=field_size)
                    self.handling_source_found = True
                    action = self.actions.pop(0)
                elif len(self.actions)  == 0:
                    # print('--- reach destination, reset detector ---')
                    self.source_detector.reset()
                    self.source_found = self.handling_source_found = False
                else: 
                    # print('--- get next action from traj ---')
                    action = self.actions.pop(0)                
        return action

# -------------------------------
# LAWN-MOWING AGENT
# -------------------------------

# [initial_position, y-direction, x-direction] 
LAWNMOWING_MODE_PRESETS = {
    'top-left-corner': {'start': (0, 0), 'main-offset': 1, 'cross-offset': 1},          # down (initially), right
    'top-right-corner': {'start': (0, 1), 'main-offset': 1, 'cross-offset': -1},        # down (initially), left
    'bottom-left-corner': {'start': (1, 0), 'main-offset': -1, 'cross-offset': 1},      # up (initially), right
    'bottom-right-corner': {'start': (1, 1), 'main-offset': -1, 'cross-offset': -1}     # up (initially), left
}
LAWNMOWING_MAIN_AXIS = 0
LAWNMOWING_CROSS_AXIS = 1

LAWNMOWING_DEFAULT_PARAMS = {
    'MODE': 'top-left-corner', # must be one of field corners
    'FIELD_SIZE': 100,
    'HALF_SCOPE_SIZE': 5,
}
class LawnMowingAgent(BaseAgent):
    def __init__(self, params=LAWNMOWING_DEFAULT_PARAMS):
        # load parameters
        self.mode = params['MODE']
        self.field_size, self.half_scope_size = params['FIELD_SIZE'], params['HALF_SCOPE_SIZE']
        presets = LAWNMOWING_MODE_PRESETS[self.mode]
        self.start_position = bounded_vec(np.array(presets['start']) * self.field_size, \
            field_size=self.field_size, padding=self.half_scope_size)
        self.main_axis, self.cross_axis = LAWNMOWING_MAIN_AXIS, LAWNMOWING_CROSS_AXIS
        self.main_direction, self.cross_direction = presets['main-offset'], presets['cross-offset']
        # initialize action_queue
        self.actions = []
        self.is_main_direction = True
        self.start_position_visited = False

    def get_next_action(self, state, params):
        position = np.array(params['position'])
        if len(self.actions) == 0:
            target = self.get_target_position(position)
            self.actions = generate_agent_trajectory(start=tuple(position), end=tuple(target), field_size=self.field_size)
        action = self.actions.pop(0)
        return action

    def get_target_position(self, position):
        self.start_position_visited = self.start_position_visited or tuple(position) == tuple(self.start_position)
        if not self.start_position_visited:
            return self.start_position
        else:
            target_position = np.array(position)
            if self.is_main_direction:
                target_position[self.main_axis] += self.main_direction * self.field_size
                self.main_direction = -self.main_direction # main direction is toogled after being used
                self.is_main_direction = False
            else:
                target_position[self.cross_axis] += self.cross_direction * self.half_scope_size * 2
                self.is_main_direction = True
            target_position = bounded_vec(target_position, field_size=self.field_size, padding=self.half_scope_size)
            return target_position

# -------------------------------
# RANDOM AGENT
# -------------------------------
class RandomAgent(BaseAgent):
    
    def __init__(self, params={}):
        pass
    
    def get_next_action(self, state, params):
        action_probs = np.array(params['action_probs'])
        possible_actions = (action_probs > 0).nonzero()[0]
        action = np.random.choice(possible_actions)
        return action