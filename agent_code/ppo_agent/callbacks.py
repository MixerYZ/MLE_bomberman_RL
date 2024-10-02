import logging
import numpy as np
import os
import torch
from .features import state_to_features 
from .train import PPO, ACTIONS, LR_ACTOR, LR_CRITIC, GAMMA, K_EPOCHS, EPS_CLIP 
import events as e

def save_model(self):
    torch.save(self.model.state_dict(), "my-saved-model.pt")

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that act(...) can be called.
    """

    def initialize_model():
        return PPO(
            state_dim=588,
            action_dim=len(ACTIONS),
            lr_actor=LR_ACTOR,
            lr_critic=LR_CRITIC,
            gamma=GAMMA,
            k_epochs=K_EPOCHS,
            eps_clip=EPS_CLIP
        )

    if self.train and not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = initialize_model()
    else:
        self.logger.info("Loading model from saved state.")
        self.model = initialize_model()
        self.model.load_state_dict(torch.load("my-saved-model.pt", weights_only=True))  # 加载模型参数

    # 初始化事件统计字典
    self.event_counts = {
        'STAYED_TOO_LONG': 0,
        'CLOSER_TO_COIN': 0,
        'FURTHER_FROM_COIN': 0,
        'OLD_AREA_REVISITED': 0,
        'NEW_AREA_EXPLORED': 0,
        'MOVED_AWAY_FROM_BOMB': 0,
        'IN_EXPLOSION_RANGE': 0,
        e.INVALID_ACTION: 0,
        e.COIN_COLLECTED: 0,
        e.KILLED_SELF: 0,
        e.BOMB_DROPPED: 0,
        e.GOT_KILLED: 0,
        e.SURVIVED_ROUND: 0,
        e.CRATE_DESTROYED: 0
    }

    # 配置日志记录器
    logging.basicConfig(level=logging.INFO)
    self.logger = logging.getLogger(__name__)

    # 初始化位置、炸弹相关和动作记录
    self.last_positions = []
    self.action_sources = []
    self.coverage_map = np.zeros((17, 17))
    self.bomb_placed = False
    self.bomb_position = None
    self.bomb_place_position = None

    # 绑定 _save_action_sources 方法到 self 对象
    self._save_action_sources = lambda: save_action_sources(self)

def save_action_sources(self):
    with open("action_sources.txt", "a") as f:
        f.write(f"{self.action_sources[-1]}\n")

def act(self, game_state: dict) -> str:
    # 在每个回合的第一个时间步进行初始化
    if game_state['step'] == 1:
        self.bomb_cooldown = 10  # 炸弹冷却时间，单位为步数
        self.last_bomb_time = -self.bomb_cooldown  # 初始化冷却时间
        self.logger.info("回合开始，初始化冷却时间")
    

    self_x, self_y = game_state['self'][3]
    arena = game_state['field'].copy()  
    current_step = game_state['step']  


    if current_step - self.last_bomb_time < self.bomb_cooldown:


        action_probs = self.model.actor(torch.tensor(state_to_features(game_state), dtype=torch.float32).unsqueeze(0)).detach().numpy().flatten()
        action_probs[ACTIONS.index('BOMB')] = 0 
        action_probs[ACTIONS.index('WAIT')] = 0 
        action_probs /= action_probs.sum() 
        

        action = np.random.choice(ACTIONS, p=action_probs)

    else:

        if self.last_positions and self.last_positions[-1] == (self_x, self_y):
            self.stay_count += 1
        else:
            self.stay_count = 0
        self.last_positions.append((self_x, self_y))


        if len(self.last_positions) > 5:
            self.last_positions.pop(0)


        crate_nearby = any(
            arena[self_x + dx, self_y + dy] not in [0, -2]
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]
        )
        others = game_state['others']
        enemy_nearby = any(
            abs(self_x - x) + abs(self_y - y) <= 1
            for _, _, _, (x, y) in others
        )

        should_place_bomb = crate_nearby or enemy_nearby

        bomb_threshold = 0.2

        if should_place_bomb:

            bomb_position = (self_x, self_y)
            explosion_range = calculate_explosion_range(bomb_position, arena)

            for (ex, ey) in explosion_range:
                arena[ex, ey] = -2

            def can_escape(x, y):

                neighbors = [(x + dx, y + dy) for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]]

                safe_directions = [(nx, ny) for nx, ny in neighbors if arena[nx, ny] in [0, -2]]


                for direction in safe_directions:
                    nx, ny = direction

                    next_neighbors = [(nx + dx, ny + dy) for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
                    for next_nx, next_ny in next_neighbors:
                        if arena[nx, ny] == -2 and arena[next_nx, next_ny] == 0:
                            return True
                        elif arena[nx, ny] == 0 and arena[next_nx, next_ny] == 0:
                            return True

                return False

            if can_escape(self_x, self_y):
                action_probs = self.model.actor(torch.tensor(state_to_features(game_state), dtype=torch.float32).unsqueeze(0)).detach().numpy().flatten()
                bomb_prob = action_probs[ACTIONS.index('BOMB')]
                
                if bomb_prob < bomb_threshold:

                    action_probs[ACTIONS.index('BOMB')] = 0 
                    action_probs[ACTIONS.index('WAIT')] = 0 
                    action_probs /= action_probs.sum() 
                    action = np.random.choice(ACTIONS, p=action_probs)
                    return action

                self.last_bomb_time = current_step  
                self.bomb_place_position = (self_x, self_y) 
                return 'BOMB'
            else:

                action_probs = self.model.actor(torch.tensor(state_to_features(game_state), dtype=torch.float32).unsqueeze(0)).detach().numpy().flatten()
                action_probs[ACTIONS.index('BOMB')] = 0 
                action_probs[ACTIONS.index('WAIT')] = 0 
                action_probs /= action_probs.sum() 
                action = np.random.choice(ACTIONS, p=action_probs)

        else:

            action_probs = self.model.actor(torch.tensor(state_to_features(game_state), dtype=torch.float32).unsqueeze(0)).detach().numpy().flatten()
            action_probs[ACTIONS.index('BOMB')] = 0 
            action_probs[ACTIONS.index('WAIT')] = 0 
            action_probs /= action_probs.sum() 
            action = np.random.choice(ACTIONS, p=action_probs)


    new_x, new_y = self_x, self_y 
        new_y += 1
    elif action == 'DOWN':
        new_y -= 1
    elif action == 'LEFT':
        new_x -= 1
    elif action == 'RIGHT':
        new_x += 1


    if arena[new_x, new_y] != 0:  

        random_prob = 0.5 
        if np.random.rand() < random_prob:
            available_actions = [a for a in ACTIONS if 
                        (a == 'DOWN' and arena[self_x, self_y + 1] == 0) or
                        (a == 'UP' and arena[self_x, self_y - 1] == 0) or
                        (a == 'LEFT' and arena[self_x - 1, self_y] == 0) or
                        (a == 'RIGHT' and arena[self_x + 1, self_y] == 0)]
            if available_actions:
                action = np.random.choice(available_actions)
                self.action_sources.append("Random for invalid action")
                self._save_action_sources()


    loop_detected = False
    if len(self.last_positions) >= 4:
        loop_patterns = [
            [(self_x, self_y), (self_x + 1, self_y), (self_x, self_y), (self_x - 1, self_y)],
            [(self_x, self_y), (self_x - 1, self_y), (self_x, self_y), (self_x + 1, self_y)],
            [(self_x, self_y), (self_x, self_y + 1), (self_x, self_y), (self_x, self_y - 1)],
            [(self_x, self_y), (self_x, self_y - 1), (self_x, self_y), (self_x, self_y + 1)]
        ]
        if self.last_positions[-4:] in loop_patterns:
            loop_detected = True
            self.logger.debug("Detected looping pattern.")


    if loop_detected or self.stay_count > 3:
        random_prob = 0.5  
        if np.random.rand() < random_prob:
            self.logger.debug("Choosing action purely at random due to loop or staying too long.")
            action = np.random.choice(ACTIONS, p=[.25, .25, .25, .25, .0, .0])
            self._save_action_sources()
            return action

    
    random_prob = 0.1 
    if self.train and np.random.rand() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=[0.25, 0.25, 0.25, 0.25, 0, 0])
    self.logger.info(f"Selected action: {action}")
    self.action_sources.append("Model")
    self._save_action_sources()
    return action

def calculate_explosion_range(bomb_position, arena, explosion_radius=3):
    x, y = bomb_position
    explosion_range = set()


    explosion_range.add((x, y))

    for i in range(1, explosion_radius + 1):
        if arena[x - i, y] == -1:
            break
        explosion_range.add((x - i, y))
    
    for i in range(1, explosion_radius + 1):
        if arena[x + i, y] == -1:
            break
        explosion_range.add((x + i, y))
    
    for i in range(1, explosion_radius + 1):
        if arena[x, y - i] == -1:
            break
        explosion_range.add((x, y - i))

    for i in range(1, explosion_radius + 1):
        if arena[x, y + i] == -1:
            break
        explosion_range.add((x, y + i))

    return explosion_range