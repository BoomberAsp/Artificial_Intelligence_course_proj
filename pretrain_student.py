import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os

# 导入 DQN 架构
from agents.cartpole_dqn import DQNSolver, DQNConfig


def pretrain_student(epochs=50, batch_size=64, lr=1e-3):
    # 1. 加载数据
    data_path = "data/expert_data.pt"
    if not os.path.exists(data_path):
        print(f"错误：找不到数据文件 {data_path}，请先运行 generate_data.py")
        return

    data = torch.load(data_path)
    states = data['states']
    actions = data['actions']

    print(f"加载了 {len(states)} 条专家数据")

    dataset = TensorDataset(states, actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. 准备学生
    cfg = DQNConfig(lr=lr)


    student_agent = DQNSolver(4, 2, cfg=cfg)

    network = student_agent.online

    network.train()  # 开启训练模式

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=lr)

    print(f"开始预训练 (Imitation Learning)... 设备: {student_agent.device}")

    # 3. 开始刷题
    for epoch in range(1, epochs + 1):
        total_loss = 0
        correct = 0
        total = 0

        for batch_states, batch_actions in dataloader:
            batch_states = batch_states.to(student_agent.device)
            batch_actions = batch_actions.to(student_agent.device)

            # 前向传播
            q_values = network(batch_states)

            # 计算 Loss (让 DQN 的输出去拟合老师的动作)
            loss = criterion(q_values, batch_actions)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计准确率
            total_loss += loss.item()
            _, predicted = torch.max(q_values, 1)
            total += batch_actions.size(0)
            correct += (predicted == batch_actions).sum().item()

        acc = 100 * correct / total
        avg_loss = total_loss / len(dataloader)

        # 每 5 轮打印一次进度
        if epoch % 5 == 0:
            print(f"Epoch [{epoch}/{epochs}] Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")

    # 4. 保存预训练模型
    # 这一步会自动把训练好的 self.online 保存下来
    save_path = "models/pretrained_dqn.torch"
    os.makedirs("models", exist_ok=True)
    student_agent.save(save_path)

    print(f"\n预训练完成！模型已保存至: {save_path}")
    print(">>> 下一步：请修改 train.py，让它加载这个模型进行后续训练。")


if __name__ == "__main__":
    pretrain_student()