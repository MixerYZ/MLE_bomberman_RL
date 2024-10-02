import matplotlib.pyplot as plt
import numpy as np
import pickle
from collections import namedtuple, deque
from typing import List
import events as e
from .callbacks import state_to_features
from .callbacks import act
from .callbacks import NUMBER_COINS, ACTIONS, ACTION_LOOPS, POSITION_MIN, POSITION_MAX
from .useful_func import get_info_from_game_state, get_valid_actions, n_closest_coins, closest_coin, Manhattan_dist, enter_bay



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify 
TRANSITION_HISTORY_SIZE = 10  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...




# Events 
VALID_ACTION = "VALID_ACTION"
INVALID_ACTION = "INVALID_ACTION"
GO_TO_COINS = "GO_TO_COINS"
AWAY_FROM_COINS = "AWAY_FROM_COINS"
PASS_BY_COIN = "PASS_BY_COIN"
ACTION_LOOP_2 = "ACTION_LOOP_2"
GO_TO_OTHER = "GO_TO_OTHER"
AWAY_FROM_OTHER = "AWAY_FROM_OTHER"
ENTER_BAY = "ENTER_BAY"

CANNOT_MOVE = "CANNOT_MOVE"
CANNOT_MOVE_2 = "CANNOT_MOVE_2"
CANNOT_MOVE_3 = "CANNOT_MOVE_3"
AWAY_FROM_CANNOT_MOVE = "AWAY_FROM_CANNOT_MOVE"
AWAY_FROM_CANNOT_MOVE_2 = "AWAY_FROM_CANNOT_MOVE_2"
AWAY_FROM_CANNOT_MOVE_3 = "AWAY_FROM_CANNOT_MOVE_3"

IN_BOMB_DANGER = "IN_BOMB_DANGER"
ENTER_BOMB_DANGER = "ENTER_BOMB_DANGER"
IN_BOMB_SAVE = "IN_BOMB_SAVE"
OTHER_NEARBY_BOMB = "OTHER_NEARBY_BOMB"




def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    old_arena, old_score, old_bombs_left, old_position, old_bombs, old_bomb_xys, old_others, old_coins, old_bomb_map = get_info_from_game_state(old_game_state)

    old_coin_dist = min([dist for (coin, dist) in n_closest_coins(old_position, old_coins, NUMBER_COINS)])
    old_valid_actions = get_valid_actions(self, old_game_state, old_arena, old_bombs_left, old_position, old_bomb_xys, old_others, old_bomb_map)
    
    new_arena, new_score, new_bombs_left, new_position, new_bombs, new_bomb_xys, new_others, new_coins, new_bomb_map = get_info_from_game_state(new_game_state)
    new_coin_dist = min([dist for (coin, dist) in n_closest_coins(new_position, new_coins, NUMBER_COINS)])
    new_valid_actions = get_valid_actions(self, new_game_state, new_arena, new_bombs_left, new_position, new_bomb_xys, new_others, new_bomb_map)
    
    action = self_action
    evaluator = self.model
    
    
    # Idea: Add your own events to hand out rewards 
    if action in old_valid_actions:
        events.append(VALID_ACTION)
    else: 
        events.append(INVALID_ACTION)

    # previous_positions = [(int(16 * gs[0][289]), int(16 * gs[0][290])) for gs in self.transitions]
    previous_positions = [(int(16 * gs[0][0]), int(16 * gs[0][1])) for gs in self.transitions]
    if new_position in previous_positions: 
        if len(previous_positions)>2 and new_position in previous_positions[-3:-1]: 
            events.append(ACTION_LOOP_2)
    
    exist_coin = bool(len(old_coins))
    n_others = len(new_others)
    exist_other = bool(n_others)
    
    
    if exist_coin: 
        if len(new_coins) < len(old_coins) or new_coin_dist < old_coin_dist: 
            events.append(GO_TO_COINS)
        else:
            events.append(AWAY_FROM_COINS)
            if old_coin_dist == 1: 
                events.append(PASS_BY_COIN)

    if not exist_other: 
        if exist_coin: 
            if len(new_coins) < len(old_coins) or new_coin_dist < old_coin_dist: 
                events.append(GO_TO_COINS)
            else:
                events.append(AWAY_FROM_COINS)
                if old_coin_dist == 1: 
                    events.append(PASS_BY_COIN)
    

    for i in range(n_others): 
        if Manhattan_dist(new_position, new_others[i]) < Manhattan_dist(old_position, old_others[i]): 
            events.append(GO_TO_OTHER)
        else: 
            events.append(AWAY_FROM_OTHER)
            
    
    
    whether_enter_bay = enter_bay(new_arena, new_position, new_bomb_xys)
    if whether_enter_bay==1:
        events.append(ENTER_BAY)
    elif whether_enter_bay==-1:
        events.append(CANNOT_MOVE)
    else: 
        (x, y) = new_position
        new_direction_coor = [(x, y-1), (x+1, y), (x, y+1), (x-1, y)]
        for bomb in new_bomb_xys: 
            if bomb==new_direction_coor[0] and action=='DOWN': 
                if new_arena[x, y+1]==0 and POSITION_MIN <= y+2 <= POSITION_MAX and new_arena[x-1, y]!=0 and new_arena[x+1, y]!=0: 
                    if enter_bay(new_arena, (x,y+1), new_bomb_xys)==1: 
                        events.append(CANNOT_MOVE_2)
                    elif POSITION_MIN <= y+3 <= POSITION_MAX and POSITION_MIN <= x-1 and x+1 <= POSITION_MAX: 
                        if new_arena[x-1, y+1]!=0 and new_arena[x+1, y+1]!=0: 
                            if np.abs(new_arena[x, y:y+2]).sum()==0 and enter_bay(new_arena, (x,y+2), new_bomb_xys)==1: 
                                events.append(CANNOT_MOVE_3)
            elif bomb==new_direction_coor[1] and action=='LEFT': 
                if new_arena[x-1, y]==0 and POSITION_MIN <= x-2 <= POSITION_MAX and new_arena[x, y-1]!=0 and new_arena[x, y+1]!=0: 
                    if enter_bay(new_arena, (x-1,y), new_bomb_xys)==1: 
                        events.append(CANNOT_MOVE_2)
                    elif POSITION_MIN <= x-3 <= POSITION_MAX and POSITION_MIN <= y-1 and y+1 <= POSITION_MAX: 
                        if new_arena[x-1, y-1]!=0 and new_arena[x-1, y+1]!=0: 
                            if np.abs(new_arena[x-2, y]).sum()==0 and enter_bay(new_arena, (x-2,y), new_bomb_xys)==1: 
                                events.append(CANNOT_MOVE_3)
            elif bomb==new_direction_coor[2] and action=='UP': 
                if new_arena[x, y-1]==0 and POSITION_MIN <= y-2 <= POSITION_MAX and new_arena[x-1, y]!=0 and new_arena[x+1, y]!=0: 
                    if enter_bay(new_arena, (x,y-1), new_bomb_xys)==1: 
                        events.append(CANNOT_MOVE_2)
                    elif POSITION_MIN <= y-3 <= POSITION_MAX and POSITION_MIN <= x-1 and x+1 <= POSITION_MAX: 
                        if new_arena[x-1, y-1]!=0 and new_arena[x+1, y-1]!=0: 
                            if np.abs(new_arena[x, y:y-2]).sum()==0 and enter_bay(new_arena, (x,y-2), new_bomb_xys)==1: 
                                events.append(CANNOT_MOVE_3)
            elif bomb==new_direction_coor[3] and action=='RIGHT': 
                if new_arena[x+1, y]==0 and POSITION_MIN <= x+2 <= POSITION_MAX and new_arena[x, y-1]!=0 and new_arena[x, y+1]!=0: 
                    if enter_bay(new_arena, (x+1,y), new_bomb_xys)==1: 
                        events.append(CANNOT_MOVE_2)
                    elif POSITION_MIN <= x+3 <= POSITION_MAX and POSITION_MIN <= y-1 and y+1 <= POSITION_MAX: 
                        if new_arena[x+1, y-1]!=0 and new_arena[x+1, y+1]!=0: 
                            if np.abs(new_arena[x+2, y]).sum()==0 and enter_bay(new_arena, (x+2,y), new_bomb_xys)==1: 
                                events.append(CANNOT_MOVE_3)

    if old_position in old_bomb_xys and CANNOT_MOVE not in events and CANNOT_MOVE_2 not in events and CANNOT_MOVE_3 not in events: 
        if action != 'DOWN' and POSITION_MIN <= old_position[1]+2 <= POSITION_MAX: 
            (x, y) = (old_position[0], old_position[1]+1)
            new_direction_coor = [(x, y-1), (x+1, y), (x, y+1), (x-1, y)]
            if enter_bay(new_arena, (x, y), new_bomb_xys) == -1: 
                events.append(AWAY_FROM_CANNOT_MOVE)
            elif new_arena[x, y+1]==0 and POSITION_MIN <= y+2 <= POSITION_MAX and new_arena[x-1, y]!=0 and new_arena[x+1, y]!=0: 
                if enter_bay(new_arena, (x,y+1), new_bomb_xys)==1: 
                    events.append(AWAY_FROM_CANNOT_MOVE_2)
                elif POSITION_MIN <= y+3 <= POSITION_MAX and POSITION_MIN <= x-1 and x+1 <= POSITION_MAX: 
                    if new_arena[x-1, y+1]!=0 and new_arena[x+1, y+1]!=0: 
                        if np.abs(new_arena[x, y:y+2]).sum()==0 and enter_bay(new_arena, (x,y+2), new_bomb_xys)==1: 
                            events.append(AWAY_FROM_CANNOT_MOVE_3)
        
        if action != 'LEFT' and POSITION_MIN <= old_position[0]-2 <= POSITION_MAX: 
            (x, y) = (old_position[0]-1, old_position[1])
            new_direction_coor = [(x, y-1), (x+1, y), (x, y+1), (x-1, y)]
            if enter_bay(new_arena, (x, y), new_bomb_xys) == -1: 
                events.append(AWAY_FROM_CANNOT_MOVE)
            elif new_arena[x-1, y]==0 and POSITION_MIN <= x-2 <= POSITION_MAX and new_arena[x, y-1]!=0 and new_arena[x, y+1]!=0: 
                if enter_bay(new_arena, (x-1,y), new_bomb_xys)==1: 
                    events.append(AWAY_FROM_CANNOT_MOVE_2)
                elif POSITION_MIN <= x-3 <= POSITION_MAX and POSITION_MIN <= y-1 and y+1 <= POSITION_MAX: 
                    if new_arena[x-1, y-1]!=0 and new_arena[x-1, y+1]!=0: 
                        if np.abs(new_arena[x-2, y]).sum()==0 and enter_bay(new_arena, (x-2,y), new_bomb_xys)==1: 
                            events.append(AWAY_FROM_CANNOT_MOVE_3)
        
        if action != 'UP' and POSITION_MIN <= old_position[1]-2 <= POSITION_MAX: 
            (x, y) = (old_position[0], old_position[1]-1)
            new_direction_coor = [(x, y-1), (x+1, y), (x, y+1), (x-1, y)]
            if enter_bay(new_arena, (x, y), new_bomb_xys) == -1: 
                events.append(AWAY_FROM_CANNOT_MOVE)
            elif new_arena[x, y-1]==0 and POSITION_MIN <= y-2 <= POSITION_MAX and new_arena[x-1, y]!=0 and new_arena[x+1, y]!=0: 
                if enter_bay(new_arena, (x,y-1), new_bomb_xys)==1: 
                    events.append(AWAY_FROM_CANNOT_MOVE_2)
                elif POSITION_MIN <= y-3 <= POSITION_MAX and POSITION_MIN <= x-1 and x+1 <= POSITION_MAX: 
                    if new_arena[x-1, y-1]!=0 and new_arena[x+1, y-1]!=0: 
                        if np.abs(new_arena[x, y:y-2]).sum()==0 and enter_bay(new_arena, (x,y-2), new_bomb_xys)==1: 
                            events.append(AWAY_FROM_CANNOT_MOVE_3)
        elif action != 'RIGHT' and POSITION_MIN <= old_position[0]+2 <= POSITION_MAX: 
            (x, y) = (old_position[0]+1, old_position[1])
            new_direction_coor = [(x, y-1), (x+1, y), (x, y+1), (x-1, y)]
            if enter_bay(new_arena, (x, y), new_bomb_xys) == -1: 
                events.append(AWAY_FROM_CANNOT_MOVE)
            elif new_arena[x+1, y]==0 and POSITION_MIN <= x+2 <= POSITION_MAX and new_arena[x, y-1]!=0 and new_arena[x, y+1]!=0: 
                if enter_bay(new_arena, (x+1,y), new_bomb_xys)==1: 
                    events.append(AWAY_FROM_CANNOT_MOVE_2)
                elif POSITION_MIN <= x+3 <= POSITION_MAX and POSITION_MIN <= y-1 and y+1 <= POSITION_MAX: 
                    if new_arena[x+1, y-1]!=0 and new_arena[x+1, y+1]!=0: 
                        if np.abs(new_arena[x+2, y]).sum()==0 and enter_bay(new_arena, (x+2,y), new_bomb_xys)==1: 
                            events.append(AWAY_FROM_CANNOT_MOVE_3)
                
    
    
    old_danger_zone = [(i, j) for i, row in enumerate(old_bomb_map) for j, val in enumerate(row) if val < 5]
    new_danger_zone = [(i, j) for i, row in enumerate(new_bomb_map) for j, val in enumerate(row) if val < 5]
    new_danger_zone.extend(old_danger_zone)
    if len(new_danger_zone) > 0: 
        if (old_position in old_danger_zone) and (new_position in new_danger_zone): 
            events.append(IN_BOMB_DANGER)
        elif (old_position not in old_danger_zone) and (new_position in new_danger_zone): 
            events.append(ENTER_BOMB_DANGER)
        else: 
            events.append(IN_BOMB_SAVE)

    if action == 'BOMB': 
        for other in old_others: 
            if Manhattan_dist(old_position, other) < 3: 
                events.append(OTHER_NEARBY_BOMB)
        for other in new_others: 
            if Manhattan_dist(old_position, other) < 3: 
                events.append(OTHER_NEARBY_BOMB)
        
    
    

    reward = reward_from_events(self, events)    
    DISCOUNT = evaluator.discount
    next_action = act(self, new_game_state)
    target = reward + DISCOUNT * evaluator.value(new_game_state, next_action)
    evaluator.learn(old_game_state, action, target)
    self.model.one_round_rewards.append(reward)
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward))

    for event in events:
        if event in self.event_counts:
            self.event_counts[event] += 1
    #self.logger.info(f"Event counts: {self.event_counts}")

    
def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    arena, score, bombs_left, (x, y), bombs, bomb_xys, others, coins, bomb_map = get_info_from_game_state(last_game_state)
    position = (x, y)
    step = last_game_state['step']
    evaluator = self.model
    reward = reward_from_events(self, events)
    
    self.model.one_round_rewards.append(reward)
    self.model.average_rewards.append(np.mean(self.model.one_round_rewards))
    self.model.one_round_rewards = []
    
    suicide = 0
    num_coins = len(coins) + int(e.COIN_COLLECTED in events)
    final_score = score + int(e.COIN_COLLECTED in events)
    if e.KILLED_SELF in events: 
        suicide = 1
       #print(events, len(others), final_score, step)
    # elif e.SURVIVED_ROUND in events: 
        #print(e.SURVIVED_ROUND, len(others), final_score, step)
    
    self.model.statistics.append([num_coins, len(others), step, final_score, suicide])
    
    valid_actions = get_valid_actions(self, last_game_state, arena, bombs_left, position, bomb_xys, others, bomb_map)
    if last_action in valid_actions:
        events.append(VALID_ACTION)
    else: 
        events.append(INVALID_ACTION)
    
    previous_positions = [(int(16 * gs[0][0]), int(16 * gs[0][1])) for gs in self.transitions]
    if position in previous_positions: 
        if len(previous_positions)>2 and position in previous_positions[-3:-1]: 
            events.append(ACTION_LOOP_2)

    whether_enter_bay = enter_bay(arena, position, bomb_xys)
    if whether_enter_bay==1:
        events.append(ENTER_BAY)
    elif whether_enter_bay==-1:
        events.append(CANNOT_MOVE)
    
    # DISCOUNT = evaluator.discount
    target = reward
    evaluator.learn(last_game_state, last_action, target)

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))
    
    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)
    

    for event in events:
        if event in self.event_counts:
            self.event_counts[event] += 1
    self.logger.info(f"Event counts: {self.event_counts}")
    # with open("my-saved-model_test.pt", "wb") as file:
    #     pickle.dump(self.model, file)
    

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        VALID_ACTION: -1, 
        INVALID_ACTION: -5, 
        e.WAITED: -10,  
        e.INVALID_ACTION: -5, 
        ACTION_LOOP_2: -20, 

        e.COIN_COLLECTED: 20,
        GO_TO_COINS: 4, 
        AWAY_FROM_COINS: -8, 
        PASS_BY_COIN: -7, 
        
        e.BOMB_DROPPED: 5, 
        e.BOMB_EXPLODED: -1, 
        e.CRATE_DESTROYED: 1, 
        e.COIN_FOUND: 2, 
        e.KILLED_OPPONENT: 20, 
        e.KILLED_SELF: -20, 
        e.GOT_KILLED: -20, 
        e.SURVIVED_ROUND: 10, 
        e.OPPONENT_ELIMINATED: 5, 
        GO_TO_OTHER: 1, 
        AWAY_FROM_OTHER: -1, 
        ENTER_BAY: 5, 
        CANNOT_MOVE: -10, 
        CANNOT_MOVE_2: -35, 
        CANNOT_MOVE_3: -35, 
        AWAY_FROM_CANNOT_MOVE: 50, 
        AWAY_FROM_CANNOT_MOVE_2: 50, 
        AWAY_FROM_CANNOT_MOVE_3: 50, 
        IN_BOMB_DANGER: -10, 
        ENTER_BOMB_DANGER: -20, 
        IN_BOMB_SAVE: 4, 
        OTHER_NEARBY_BOMB: 2 
    }
    
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            # print(event)
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    # print(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
