import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import itertools
import events as e

from .tile_coding import IHT, hash_coords, tiles
from .traces import accumulating_trace, replacing_trace, replacing_trace_with_clearing, dutch_trace 
from .useful_func import get_info_from_game_state, get_valid_actions, n_closest_coins, closest_coin, Manhattan_dist, Boltzmann, get_escape_route

# bound for position
POSITION_MIN = 0
POSITION_MAX = 16
NUMBER_COINS = 1
TEMPERATURE = 401

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_LOOPS = [('UP', 'DOWN'), ('DOWN', 'UP'), ('RIGHT', 'LEFT'), ('LEFT', 'RIGHT'), ('BOMB', 'BOMB')]


# wrapper class for Sarsa(lambda)
class Sarsa:
    # In this example I use the tiling software instead of implementing standard tiling by myself
    # One important thing is that tiling is only a map from (state, action) to a series of indices
    # It doesn't matter whether the indices have meaning, only if this map satisfy some property 
    # @maxSize: the maximum # of indices
    def __init__(self, step_size, lam, trace_update=accumulating_trace, num_of_tilings=8, max_size=2048):
        self.max_size = max_size
        self.num_of_tilings = num_of_tilings
        self.trace_update = trace_update
        self.lam = lam

        # divide step size equally to each tiling
        self.step_size = step_size / num_of_tilings

        self.hash_table = IHT(max_size)

        # weight for each tile
        # self.weights = np.zeros(max_size)
        self.weights = np.random.uniform(size=max_size) * 0.01

        # trace for each tile
        self.trace = np.zeros(max_size)

        # parameters 
        self.epsilon = 0.2
        self.discount = 0.9
        
        self.one_round_rewards = []
        self.average_rewards = []
        self.statistics = []

    # get indices of active tiles for given state and action
    def get_active_tiles(self, game_state, action):
        normalized_floats = state_to_features(game_state)
        normalized_floats = [self.num_of_tilings * f for f in normalized_floats]
        active_tiles = tiles(self.hash_table, self.num_of_tilings, normalized_floats, [action])
        return active_tiles

    # estimate the value of given state and action
    def value(self, game_state, action):
        active_tiles = self.get_active_tiles(game_state, action)
        value = np.sum(self.weights[active_tiles])
        return value

    # learn with given state, action and target
    def learn(self, game_state, action, target):
        active_tiles = self.get_active_tiles(game_state, action)
        estimation = np.sum(self.weights[active_tiles])
        delta = target - estimation
        if self.trace_update == accumulating_trace or self.trace_update == replacing_trace:
            self.trace_update(self.trace, active_tiles, self.lam)
        elif self.trace_update == dutch_trace:
            self.trace_update(self.trace, active_tiles, self.lam, self.step_size)
        elif self.trace_update == replacing_trace_with_clearing:
            clearing_tiles = []
            for act in ACTIONS:
                if act != action:
                    clearing_tiles.extend(self.get_active_tiles(game_state, act))
            self.trace_update(self.trace, active_tiles, self.lam, clearing_tiles)
        else:
            raise Exception('Unexpected Trace Type')
        self.weights += self.step_size * delta * self.trace
        

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.previous_action = None
    self.previous_bomb_map = None
    if self.train and not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from Sarsa.")
        print("Setting up model from Sarsa.")
        alpha = 0.9
        lam = 0.9
        trace = dutch_trace # accumulating_trace, replacing_trace, replacing_trace_with_clearing, dutch_trace 
        self.model = Sarsa(alpha, lam, trace, max_size=409600)
    else:
        self.logger.info("Loading model from saved state.")
        print("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)

            # plt.plot(self.model.average_rewards)
            # plt.xlabel('Index')
            # plt.ylabel('Value')
            # plt.title('Line Plot of List Values')
            # plt.show()
            # print(len(self.model.average_rewards))
        # 初始化事件统计字典
    self.event_counts = {
        # 'STAYED_TOO_LONG': 0,
        # 'CLOSER_TO_COIN': 0,
        # 'FURTHER_FROM_COIN': 0,
        # 'OLD_AREA_REVISITED': 0,
        # 'NEW_AREA_EXPLORED': 0,
        # 'MOVED_AWAY_FROM_BOMB': 0,
        # 'IN_EXPLOSION_RANGE': 0,
        e.INVALID_ACTION: 0,
        e.COIN_COLLECTED: 0,
        e.KILLED_SELF: 0,
        e.BOMB_DROPPED: 0,
        e.GOT_KILLED: 0,
        e.SURVIVED_ROUND: 0,
        e.CRATE_DESTROYED: 0,
        e.KILLED_OPPONENT: 0
    }


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """    
    arena, score, bombs_left, (x, y), bombs, bomb_xys, others, coins, bomb_map = get_info_from_game_state(game_state)
    step = game_state['step']
    position = (x, y)

    exist_bombs = bool(len(bombs))
    exist_crates = bool(np.max(arena))
    exist_others = bool(len(others))
    exist_coins = bool(len(coins))

    danger_zone = [(i, j) for i, row in enumerate(bomb_map) for j, val in enumerate(row) if val < 5]
    if self.previous_bomb_map is not None: 
        danger_zone.extend([(i, j) for i, row in enumerate(self.previous_bomb_map) for j, val in enumerate(row) if val < 5])
        
    exist_danger = bool(len(danger_zone))
    
    
    if position in danger_zone: 
        in_danger_zone = True 
    else: 
        in_danger_zone = False 
    
    direction_coor = [(x, y-1), (x+1, y), (x, y+1), (x-1, y)]
    surround = [arena[d] for d in direction_coor]
    
    possible_actions = []
    if in_danger_zone: 
        for i in range(4): 
            if arena[direction_coor[i]] == 0 and direction_coor[i] not in bomb_xys: 
                possible_actions.append(ACTIONS[i])
    else: 
        for i in range(4): 
            if arena[direction_coor[i]] == 0 and direction_coor[i] not in danger_zone: 
                possible_actions.append(ACTIONS[i])
        
        if exist_bombs: 
            bomb_dists = [Manhattan_dist(position, bomb) for bomb in bomb_xys]
            if min(bomb_dists) < 5: 
                possible_actions.append('WAIT')
        # elif exist_others: 
        #     other_dists = [Manhattan_dist(position, other) for other in others]
        #     if min(other_dists) < 5: 
        #         possible_actions.append('WAIT')
            

        if exist_others or exist_crates: 
            # if exist_danger: 
            #     possible_actions.append('WAIT')

            if not in_danger_zone: 
                nearby_crate = (1 in surround)
                enter_bay = (np.abs(np.array(surround)).sum()==3)
                if enter_bay: 
                    possible_actions.append('BOMB')
                elif nearby_crate: 
                    escape_route, escape_prob = get_escape_route(arena, position, POSITION_MIN, POSITION_MAX)
                    if (self.train and escape_prob>0) or (np.random.binomial(1, escape_prob)==1 and escape_prob>3/12): 
                        # print(escape_prob)
                        possible_actions.append('BOMB')

#                     clean_roads = [0, 0, 0, 0]
#                     for k in range(1, 6): 
#                         if POSITION_MIN <= y-k <= POSITION_MAX: 
#                             clean_roads[0] += abs(arena[(x, y-k)])
#                         if POSITION_MIN <= x+k <= POSITION_MAX: 
#                             clean_roads[1] += abs(arena[(x+k, y)])
#                         if POSITION_MIN <= y+k <= POSITION_MAX: 
#                             clean_roads[2] += abs(arena[(x, y+k)])
#                         if POSITION_MIN <= x-k <= POSITION_MAX: 
#                             clean_roads[3] += abs(arena[(x-k, y)])

#                     if clean_roads[0]+clean_roads[2]==0 or clean_roads[1]+clean_roads[3]==0: 
#                         possible_actions.append('BOMB')
#                     else: 
#                         bomb_pr = min(np.array([int(not bool(tf)) for tf in clean_roads]).sum() / 3, 1)
#                         bomb_pr = max(0.05, bomb_pr)
#                         if self.train or np.random.binomial(1, bomb_pr) == 1: 
#                             possible_actions.append('BOMB')
                        
                        
                if 'BOMB' not in possible_actions: 
                    _, escape_prob = get_escape_route(arena, position, POSITION_MIN, POSITION_MAX)
                    for other in others: 
                        if Manhattan_dist(other, position)<5 and (np.random.binomial(1, escape_prob)==1 and escape_prob>3/12): 
                            # print(escape_prob)
                            possible_actions.append('BOMB')
                            break

    if len(possible_actions)==0: 
        possible_actions.append('WAIT')

    evaluator = self.model
    values = []
    # todo Exploration vs exploitation 
    EPSILON = evaluator.epsilon

    if self.train and not in_danger_zone and np.random.binomial(1, EPSILON) == 1: 
        self.logger.debug("Choosing action purely at random.")
        if len(possible_actions) > 1: 
            pr = Boltzmann(evaluator.value, game_state, possible_actions[:-1], TEMPERATURE-step)
            action = np.random.choice(possible_actions[:-1], p=pr)
            # action = np.random.choice(possible_actions[:-1])
        else: 
            pr = Boltzmann(evaluator.value, game_state, possible_actions, TEMPERATURE-step)
            action = np.random.choice(possible_actions, p=pr)
            # action = np.random.choice(possible_actions)
    else: 
        for action in possible_actions:
            v = evaluator.value(game_state, action)
            if not self.train and action == 'WAIT': 
                if v >= 0: 
                    v /= 5
                else: 
                    v *= 2
            
            values.append(v)
        action = possible_actions[np.argmax(values)]
    
    if (self.previous_action, action) in ACTION_LOOPS: 
        if len(possible_actions)>1: 
            if np.random.binomial(1, 0.8)==1 or in_danger_zone: 
                possible_actions.remove(action)
        
        pr = Boltzmann(evaluator.value, game_state, possible_actions, TEMPERATURE-step)
        action = np.random.choice(possible_actions, p=pr)
        # action = np.random.choice(possible_actions)

        
    if not exist_bombs and not exist_crates and not exist_others and not exist_coins and not exist_danger: 
        action = 'WAIT'
    
    self.previous_action = action
    self.previous_bomb_map = bomb_map

    self.logger.debug(f'{position}, {action}, {len(others)}, {len(coins)}, {step}, {score}')
    # if not self.train: 
        # print(position, possible_actions, values, in_danger_zone, action)
    return action


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    arena, score, bombs_left, (x, y), bombs, bomb_xys, others, coins, bomb_map = get_info_from_game_state(game_state)
    step = game_state['step']
    position = (x, y)
    top_n_coins_with_dist = n_closest_coins(position, coins, NUMBER_COINS)
    top_n_coins = [coin for (coin, t) in top_n_coins_with_dist]
    while len(others) < 3: 
        others.append((0, 0))

    extended_arena = np.full((23, 23), -1)
    extended_arena[3:20, 3:20] = arena
    extended_bomb_map = np.full((23, 23), -1)
    extended_bomb_map[3:20, 3:20] = bomb_map

    feature = [position[0] / (POSITION_MAX-POSITION_MIN), position[1] / (POSITION_MAX-POSITION_MIN)]
    feature.extend(other_coord / (POSITION_MAX-POSITION_MIN) for other_coord in list(itertools.chain(*others)))
    feature.extend(coin_coord / (POSITION_MAX-POSITION_MIN) for coin_coord in list(itertools.chain(*top_n_coins)))
    feature.extend([step / 400])
    feature.extend([(a+1)/2 for a in extended_arena[x-1:x+8, y-1:y+8].reshape(-1).tolist()]) # 81 elements
    feature.extend(b / 5 for b in extended_bomb_map[x-1:x+8, y-1:y+8].reshape(-1).tolist()) # 81 elements
    
    return feature 

    # current_score = min([score, 65])
    # num_coins = min([len(coins), 50])

    # direction_coor = [(x, y-1), (x+1, y), (x, y+1), (x-1, y)]
    # feature = [(arena[d]+1)/2 for d in direction_coor]

    # feature = [(a+1)/2 for a in arena.reshape(-1).tolist()] # 289 elements 
    # feature.extend([position[0] / (POSITION_MAX-POSITION_MIN), position[1] / (POSITION_MAX-POSITION_MIN)])
    # feature.extend(other_coord / (POSITION_MAX-POSITION_MIN) for other_coord in list(itertools.chain(*others)))
    # feature.extend(coin_coord / (POSITION_MAX-POSITION_MIN) for coin_coord in list(itertools.chain(*top_n_coins)))
    # feature.extend(b / 5 for b in bomb_map.reshape(-1).tolist())
    # feature.extend([step / 400, current_score / 65, num_coins / 50])




    # # For example, you could construct several channels of equal shape, ...
    # channels = []
    # channels.append(...)
    # # concatenate them as a feature tensor (they must have the same shape), ...
    # stacked_channels = np.stack(channels)
    # # and return them as a vector
    # return stacked_channels.reshape(-1)
