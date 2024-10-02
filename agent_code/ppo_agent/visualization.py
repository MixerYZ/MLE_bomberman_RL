import matplotlib.pyplot as plt

# 读取奖励数据
with open("/Users/yuzhenhe/Projects/bomberman_rl/agent_code/my_agent/rewards.txt", "r") as f:
    rewards = [float(line.strip()) for line in f]

# 读取损失数据
policy_losses = []
value_losses = []
entropy_losses = []

with open("/Users/yuzhenhe/Projects/bomberman_rl/agent_code/my_agent/loss_log.txt", "r") as f:
    for line in f:
        try:
            policy_loss, value_loss, entropy_loss = map(float, line.strip().split(","))
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            entropy_losses.append(entropy_loss)
        except ValueError as e:
            print(f"Error processing line: {line.strip()} - {e}")

# 读取参数变化数据
parameter_changes = []
with open("/Users/yuzhenhe/Projects/bomberman_rl/agent_code/my_agent/parameters_log.txt", "r") as f:
    for line in f:
        try:
            change_value = line.strip().split(":")[-1].strip()
            parameter_changes.append(float(change_value))
        except ValueError as e:
            print(f"Error converting line to float: {line.strip()} - {e}")

# 读取梯度变化数据
gradient_changes = []
with open("/Users/yuzhenhe/Projects/bomberman_rl/agent_code/my_agent/gradients_log.txt", "r") as f:
    for line in f:
        try:
            gradient_value = line.strip().split(":")[-1].strip()
            gradient_changes.append(float(gradient_value))
        except ValueError as e:
            print(f"Error converting line to float: {line.strip()} - {e}")

# 读取动作来源数据
action_sources = []
with open("/Users/yuzhenhe/Projects/bomberman_rl/agent_code/my_agent/action_sources.txt", "r") as f:
    for line in f:
        action_sources.append(line.strip())

# 统计每个时间步的动作来源数量
model_counts = []
random_counts = []
model_count = 0
random_count = 0

for source in action_sources:
    if source == "Model":
        model_count += 1
    elif source == "Random for loop or staying":
        random_count += 1
    model_counts.append(model_count)
    random_counts.append(random_count)

# 创建图像
plt.figure(figsize=(15, 10))  # 调整图像大小

# 绘制奖励曲线
plt.subplot(2, 3, 1)
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode")

# 绘制价值损失和熵损失曲线
plt.subplot(2, 3, 2)
plt.plot(value_losses, label="Value Loss")
plt.plot(entropy_losses, label="Entropy Loss")
plt.xlabel("Update Step")
plt.ylabel("Loss")
plt.title("Value Loss and Entropy Loss per Update Step")
plt.legend()

# 单独绘制放大的策略损失曲线
plt.subplot(2, 3, 3)
plt.plot([x * 1e6 for x in policy_losses], label="Policy Loss (scaled by 1e6)")
plt.xlabel("Update Step")
plt.ylabel("Policy Loss (scaled)")
plt.title("Policy Loss per Update Step")
plt.legend()

# 绘制动作来源数量变化趋势曲线
plt.subplot(2, 3, 4)
plt.plot(model_counts, label="Model")
plt.plot(random_counts, label="Random for loop or staying")
plt.xlabel("Time Step")
plt.ylabel("Cumulative Action Source Count")
plt.title("Cumulative Action Source Count Over Time")
plt.legend()

# 绘制参数变化曲线
plt.subplot(2, 3, 5)
plt.plot(parameter_changes, label="Parameter Changes")
plt.xlabel("Update Step")
plt.ylabel("Parameter Change")
plt.title("Parameter Changes Over Time")
plt.legend()

# 绘制梯度变化曲线
plt.subplot(2, 3, 6)
plt.plot(gradient_changes, label="Gradient Norms")
plt.xlabel("Update Step")
plt.ylabel("Gradient Norm")
plt.title("Gradient Norms Over Time")
plt.legend()

# 调整布局以避免重叠
plt.tight_layout()

# 显示图像
plt.show()