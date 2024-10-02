import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
from collections import namedtuple, deque
from typing import List
import events as e
from .callbacks import state_to_features


TRANSITION_HISTORY_SIZE = 8000  # Buffer
GAMMA = 0.99 
# EPSILON = 0.5  
LR_ACTOR = 0.001 
LR_CRITIC = 0.001 
ENTROPY_BETA = 0.03  
K_EPOCHS = 4 
EPS_CLIP = 0.5 
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']  


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Transition namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)  
        )

    def forward(self, state):
        return self.actor(state)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)  
        )

    def forward(self, state):
        return self.critic(state)

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

class PPO(nn.Module):
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip):
        super(PPO, self).__init__()
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.buffer = RolloutBuffer()
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)

        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr_actor},
            {'params': self.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = Actor(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.actor.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        for epoch in range(self.k_epochs):
            action_probs = self.actor(old_states)
            dist = Categorical(action_probs)
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()

            state_values = self.critic(old_states).squeeze()
            advantages = rewards - old_state_values.detach()

            ratios = torch.exp(logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * self.MseLoss(state_values, rewards).mean()
            entropy_loss = -ENTROPY_BETA * dist_entropy.mean()

            total_loss = policy_loss + value_loss + entropy_loss

         
            with open("loss_log.txt", "a") as f:
                f.write(f"{policy_loss.item()}, {value_loss.item()}, {entropy_loss.item()}\n")
            print(f"Epoch {epoch}, Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()}, Entropy Loss: {entropy_loss.item()}")

            if epoch == 0:
                with open("gradients_log.txt", "a") as f_grad:
                    for name, param in self.actor.named_parameters():
                        if param.grad is not None:
                            f_grad.write(f"{name} - Gradient norm: {param.grad.norm().item()}\n")

            old_params = {name: param.clone() for name, param in self.actor.named_parameters()}


            self.optimizer.zero_grad()
            total_loss.backward()

            self.optimizer.step()


            with open("parameters_log.txt", "a") as f_param:
                for name, param in self.actor.named_parameters():
                    param_change = (param - old_params[name]).norm().item()
                    f_param.write(f"{name} - Parameter change: {param_change}\n")

        self.policy_old.load_state_dict(self.actor.state_dict())
        self.buffer.clear()

def setup_training(self):
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.model = PPO(state_dim=588, action_dim=len(ACTIONS), lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, gamma=GAMMA, k_epochs=K_EPOCHS, eps_clip=EPS_CLIP)
    self.total_rewards = []
    

    try:
        self.model.load_state_dict(torch.load("my-saved-model.pt"))
        self.model.eval()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("No saved model found, starting from scratch.")

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

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    

    # current_step = new_game_state['step']


    # for _ in range(current_step):
    #     events.append('LONG_SURVIVAL')


    # if current_step > 0:
    #     events.append('LONG_SURVIVAL')


    if self_action == 'BOMB':
        bomb_position = old_game_state['self'][3]  
        self.bomb_position = bomb_position  
        self.bomb_placed = True  
        self.timer = 4 
        self.explosion_duration = 2 
        

        self.explosion_range = calculate_explosion_range(bomb_position, old_game_state['field'])
        #print(f"炸弹爆炸范围: {self.explosion_range}")


    if self.bomb_placed:
        self.timer -= 1 
        
        agent_position = new_game_state['self'][3] 

        if agent_position in self.explosion_range and self.timer > 0:
            events.append('IN_EXPLOSION_RANGE')
        
        elif agent_position not in self.explosion_range and self.timer > 0:
            events.append('MOVED_AWAY_FROM_BOMB')

        if self.timer == 0:
            self.bomb_placed = False
            self.explosion_timer = self.explosion_duration  

    if hasattr(self, 'explosion_timer') and self.explosion_timer > 0:
        self.explosion_timer -= 1 
        
        agent_position = new_game_state['self'][3] 

        if agent_position in self.explosion_range:
            events.append('IN_EXPLOSION_RANGE')

        if self.explosion_timer == 0:
            del self.explosion_timer

    if old_game_state and new_game_state:
        old_position = old_game_state['self'][3]  # 'self' (id, name, score, position)
        new_position = new_game_state['self'][3]

        old_coins = old_game_state['coins']
        new_coins = new_game_state['coins']

        def closest_coin_distance(position, coins):
            if len(coins) == 0:
                return float('inf')
            return min([np.linalg.norm(np.array(position) - np.array(coin)) for coin in coins])

        old_distance = closest_coin_distance(old_position, old_coins)
        new_distance = closest_coin_distance(new_position, new_coins)

        if new_distance < old_distance:
            events.append('CLOSER_TO_COIN')
        elif new_distance > old_distance:
            events.append('FURTHER_FROM_COIN')

    if old_game_state:

        current_position = new_game_state['self'][3]  # 'self' (id, name, score, position)
        x, y = current_position  

        if self.coverage_map[x][y] == 0:
            events.append('NEW_AREA_EXPLORED')
            print("访问标记",self.coverage_map)
            self.coverage_map[x][y] = 1 
            
        else:
            events.append('OLD_AREA_REVISITED')
            #print("访问更新：",self.coverage_map) 
        
    if old_game_state and new_game_state:
        old_pos = old_game_state['self'][3]
        new_pos = new_game_state['self'][3]
        if old_pos == new_pos:
            if not hasattr(self, 'stay_count'):
                self.stay_count = 0
            self.stay_count += 1
        else:
            self.stay_count = 0

        if self.stay_count > 2:
            events.append('STAYED_TOO_LONG')

    
    state_features = state_to_features(old_game_state)
    state_tensor = torch.FloatTensor(state_features).to(device)
    
    action_probs = self.model.policy_old(state_tensor)
    dist = Categorical(action_probs)
    action = dist.sample()
    action_logprob = dist.log_prob(action)
    state_val = self.model.critic(state_tensor)

    self.model.buffer.states.append(state_tensor)
    self.model.buffer.actions.append(action)
    self.model.buffer.logprobs.append(action_logprob)
    self.model.buffer.state_values.append(state_val)

    reward = reward_from_events(self, events)
    #print("一个时间步的奖励：",reward)
    self.model.buffer.rewards.append(reward)
    self.model.buffer.is_terminals.append(new_game_state is None)

    for event in events:
        if event in self.event_counts:
            self.event_counts[event] += 1

    #print("这是当前时间步所有事件：", events)

    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    total_reward = sum(self.model.buffer.rewards)
    self.total_rewards.append(total_reward)
    #print("这是end_of_round总奖励：",total_reward)

    for event in events:
        if event in self.event_counts:
            self.event_counts[event] += 1

    #print("这是end_of_round所有事件：", events)

    with open("rewards.txt", "a") as f:
        f.write(f"{total_reward}\n")

    self.logger.info(f'Total reward for this round: {total_reward}')
    self.logger.info(f"Event counts: {self.event_counts}")
    self.model.update()
    torch.save(self.model.state_dict(), "my-saved-model.pt")


    if len(self.model.buffer.states) > TRANSITION_HISTORY_SIZE:
        self.model.buffer.clear()
        #print("缓冲区已清空，超出最大大小")

def reward_from_events(self, events: List[str]) -> int:
    game_rewards = {
        e.MOVED_LEFT: -10,
        e.MOVED_RIGHT: -10,
        e.MOVED_UP: -10,
        e.MOVED_DOWN: -10,
        e.WAITED: -50,
        e.COIN_COLLECTED: 50,
        #e.KILLED_SELF: -50,
        e.INVALID_ACTION: -10,
        #'STAYED_TOO_LONG': -10,
        'OLD_AREA_REVISITED': -50,
        'NEW_AREA_EXPLORED': 50,
        'CLOSER_TO_COIN': 10,
        'FURTHER_FROM_COIN': -10,

        #BOMB
        e.KILLED_SELF: -50,
        #e.BOMB_DROPPED: -10,
        e.GOT_KILLED: -100,
        #e.SURVIVED_ROUND: 100,
        #e.CRATE_DESTROYED: 10,
        'MOVED_AWAY_FROM_BOMB': 100,
        'IN_EXPLOSION_RANGE': -100,
        #'LONG_SURVIVAL': 10,
    }

    reward_sum = sum(game_rewards.get(event, 0) for event in events)
    return reward_sum