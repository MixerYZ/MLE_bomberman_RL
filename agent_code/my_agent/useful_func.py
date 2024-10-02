import numpy as np

# Reference: This function code comes from ’coin_collector_agent.py‘
def get_info_from_game_state(game_state): 
    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)
    
    return arena, score, bombs_left, (x, y), bombs, bomb_xys, others, coins, bomb_map


# Reference: This function code comes from ’coin_collector_agent.py‘
def get_valid_actions(self, game_state, arena, bombs_left, position, bomb_xys, others, bomb_map): 
    (x, y) = position
    # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] < 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    # if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')
    if (bombs_left > 0): valid_actions.append('BOMB')
    self.logger.debug(f'Valid actions: {valid_actions}')
    return valid_actions


##############################################################

def Manhattan_dist(p1, p2): 
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def closest_coin(position, coins): 
    if len(coins) == 0: 
        coin = position
    else: 
        coins_dists = [Manhattan_dist(c, position) for c in coins]
        dist = min(coins_dists)
        coin = coins[coins_dists.index(dist)]

    return coin

def n_closest_coins(position, coins, n): 
    coins_dists = [(coin, Manhattan_dist(position, coin)) for coin in coins]
    while len(coins_dists) < n: 
        coins_dists.append((position, 0))
        
    coins_dists.sort(key=lambda x: x[1])
    top_n_coins = coins_dists[:n]
    return top_n_coins

def Boltzmann(Q_value, game_state, actions, T): 
    numerators = [np.exp(Q_value(game_state, act) / T) for act in actions]
    denominator = np.array(numerators).sum()
    probs = [n / denominator for n in numerators]
    return probs

def enter_bay(new_arena, new_position, new_bomb_xys): 
    (x, y) = new_position
    new_direction_coor = [(x, y-1), (x+1, y), (x, y+1), (x-1, y)]
    bay = [abs(new_arena[site]) for site in new_direction_coor]
    if np.array(bay).sum() == 3: 
        if new_direction_coor[bay.index(0)] not in new_bomb_xys: 
            return 1
        else: 
            return -1
    else: 
        return 0

    
def get_escape_route(arena, position, POSITION_MIN, POSITION_MAX): 
    (x, y) = position
    clean_roads = [0, 0, 0, 0]
    for k in range(1, 6): 
        if POSITION_MIN <= y-k <= POSITION_MAX: 
            clean_roads[0] += abs(arena[(x, y-k)])
        if POSITION_MIN <= x+k <= POSITION_MAX: 
            clean_roads[1] += abs(arena[(x+k, y)])
        if POSITION_MIN <= y+k <= POSITION_MAX: 
            clean_roads[2] += abs(arena[(x, y+k)])
        if POSITION_MIN <= x-k <= POSITION_MAX: 
            clean_roads[3] += abs(arena[(x-k, y)])
    
    escape_route = [int(not bool(tf)) for tf in clean_roads]
    escape_route.extend(
    [int(np.abs(arena[x-1:x, y-1]).sum()==0), int(np.abs(arena[x:x+1, y-1]).sum()==0), 
     int(np.abs(arena[x+1, y-1:y]).sum()==0), int(np.abs(arena[x+1, y:y+1]).sum()==0), 
     int(np.abs(arena[x-1:x, y+1]).sum()==0), int(np.abs(arena[x:x+1, y+1]).sum()==0), 
     int(np.abs(arena[x-1, y-1:y]).sum()==0), int(np.abs(arena[x-1, y:y+1]).sum()==0)]
    )
    escape_prob = np.array(escape_route).sum() / len(escape_route)
    if escape_route[0]+escape_route[2]==2 or escape_route[1]+escape_route[3]==2: 
        escape_prob = 1
    return escape_route, escape_prob
