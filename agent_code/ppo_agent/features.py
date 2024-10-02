import numpy as np
import torch  # 导入 PyTorch

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def find_closest_objects(field, position, targets):
    distances = []
    for target in targets:
        dist = manhattan_distance(position, target)  # 曼哈顿距离
        distances.append((target, dist))
    
    # 按照距离排序
    distances.sort(key=lambda x: x[1])
    
    # 处理距离相等的情况
    closest_objects = []
    if distances:
        min_distance = distances[0][1]
        for pos, dist in distances:
            if dist == min_distance:
                closest_objects.append(pos)
    
    # 随机选择其中一个位置
    if closest_objects:
        print("最近金币位置", closest_objects)
        selected_object = np.random.choice(len(closest_objects))  # 随机选择一个最近的对象
        print("最近的位置", closest_objects[selected_object])
        return list(closest_objects[selected_object])  # 返回 [x, y] 格式
    else:
        return (0, 0)  # 如果没有找到对象，返回 [0, 0]

def state_to_features(game_state: dict) -> torch.Tensor:  # 修改返回类型为 torch.Tensor
    # 提取固定长度的特征
    step = game_state['step']  # 步数，标量
    field_flat = game_state['field'].flatten()  # 扁平化场地，固定长度为 289 (17x17)
    explosion_map_flat = game_state['explosion_map'].flatten()  # 扁平化爆炸范围，固定长度为 289

    # 提取 self 信息
    self_info = game_state['self']  # ('my_agent', 6, True, (x, y))
    self_x, self_y = self_info[3]  # 代理的位置
    self_score = self_info[1]  # 代理的分数
    self_position = torch.tensor([self_x, self_y], dtype=torch.float32)  # 代理位置特征

    # 处理 others 特征
    others_positions = [(_, _, _, (x, y)) for _, _, _, (x, y) in game_state.get('others', [])]
    closest_others = find_closest_objects(game_state['field'], (self_x, self_y), [(x, y) for _, _, _, (x, y) in others_positions])
    others_position = torch.tensor(closest_others, dtype=torch.float32)  # 直接使用 [x, y] 格式

    # 处理 bombs 特征
    bombs_positions = [((x, y), t) for ((x, y), t) in game_state.get('bombs', [])]
    closest_bombs = find_closest_objects(game_state['field'], (self_x, self_y), [(x, y) for (x, y), _ in bombs_positions])
    bombs_position = torch.tensor(closest_bombs, dtype=torch.float32)  # 用 [x, y] 格式

    # 处理 coins 特征
    coins_positions = [(x, y) for (x, y) in game_state.get('coins', [])]
    closest_coins = find_closest_objects(game_state['field'], (self_x, self_y), coins_positions)
    coins_position = torch.tensor(closest_coins, dtype=torch.float32)  # 用 [x, y] 格式

    # 组合特征
    stacked_features = torch.cat([
        torch.tensor([step], dtype=torch.float32),  # 步数
        torch.tensor(field_flat, dtype=torch.float32),  # 场地特征
        self_position,  # 代理位置特征
        torch.tensor([self_score], dtype=torch.float32),  # 添加代理分数
        torch.tensor(explosion_map_flat, dtype=torch.float32),  # 爆炸范围特征
        others_position,  # 最近的其他代理位置（？放全部）
        bombs_position,  # 最近的炸弹位置（？）
        coins_position  # 最近的金币位置（当前金币数量）
    ])
    
    return stacked_features  # 返回 PyTorch 张量