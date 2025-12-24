import gymnasium as gym
import numpy as np
import torch
import os
from agents.cartpole_physics import PhysicsAgent, PhysicsConfig


def generate_expert_dataset(num_samples=10000, save_path="data/expert_data.pt"):
    """
    让物理老师玩游戏，收集 (State, Action) 对。
    """
    # 1. 准备环境和老师
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # 最好参数
    best_cfg = PhysicsConfig(
        theta_coef=1.0,
        omega_coef=1.0,
        pos_coef=0.1,
        vel_coef=0.1
    )
    teacher = PhysicsAgent(obs_dim, act_dim, cfg=best_cfg)

    print(f"开始生成数据，目标样本数: {num_samples}...")

    # 2. 数据容器
    # 我们需要收集：当前状态 (Obs) -> 老师的动作 (Action)
    # 这里我们先收集基础的 (State, Action)
    collected_states = []
    collected_actions = []
   # states_and_actions = []

    state, _ = env.reset()
    total_steps = 0

    while len(collected_states) < num_samples:
        # 记录当前状态
       # state_and_action = tuple()
        collected_states.append(state)

        # 老师做决策
        action = teacher.act(state, evaluation_mode=True)
        collected_actions.append(action)

        # 执行
        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if done:
            state, _ = env.reset()
        else:
            state = next_state

        total_steps += 1
        if total_steps % 1000 == 0:
            print(f"已收集 {len(collected_states)} / {num_samples} 条数据...")

    env.close()

    # 3. 转换为 Tensor 并保存
    # states: [N, 4], actions: [N]
    data = {
        "states": torch.FloatTensor(np.array(collected_states)),
        "actions": torch.LongTensor(np.array(collected_actions))
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(data, save_path)
    print(f"\n[完成] 专家数据已保存至: {save_path}")
    print(f"数据形状: States {data['states'].shape}, Actions {data['actions'].shape}")


if __name__ == "__main__":
    # 生成 10,000 条数据（大概相当于玩 20-50 局满分游戏）
    generate_expert_dataset(num_samples=10000)