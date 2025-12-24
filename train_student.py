""""
train_student.py: 你的专属训练脚本
功能: 加载 Physics 预训练模型 -> 微调 (Imitation Learning) -> 考核
"""
import os
import numpy as np
import gymnasium as gym
import torch
import argparse
import time

# 复用 train.py 里的组件，避免重复
from agents.cartpole_dqn import DQNSolver, DQNConfig
from scores.score_logger import ScoreLogger
from train import evaluate_agent  # 直接调用通用评估函数

ENV_NAME = "CartPole-v1"
MODEL_DIR = "models"
PRETRAINED_PATH = "models/pretrained_dqn.torch"
STUDENT_FINAL_PATH = "models/student_final.torch"


def train_student_agent(num_episodes: int = 20):
    print(f"\n=== 启动学生微调 (Imitation -> RL Transfer) ===")

    if not os.path.exists(PRETRAINED_PATH):
        print(f"❌ 错误: 找不到预训练模型 {PRETRAINED_PATH}。请先运行 pretrain_student.py")
        return None

    os.makedirs(MODEL_DIR, exist_ok=True)
    env = gym.make(ENV_NAME)
    # 使用独立的 logger，生成 "CartPole-v1_student.csv"
    logger = ScoreLogger(ENV_NAME + "_student")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # 1. 实例化 Agent
    agent = DQNSolver(obs_dim, act_dim, cfg=DQNConfig())

    # 2. 加载预训练权重
    agent.load(PRETRAINED_PATH)
    print(f"[Transfer] 已加载 Physics 经验: {PRETRAINED_PATH}")

    # 3. 施加微调策略 (防止灾难性遗忘)
    # (A) 降低学习率 1e-3 -> 1e-5
    new_lr = 1e-5
    for param_group in agent.optim.param_groups:
        param_group['lr'] = new_lr

    # (B) 锁定探索率 (Epsilon) -> 0.01
    agent.exploration_rate = 0.01
    if hasattr(agent.cfg, 'eps_min'): agent.cfg.eps_min = 0.01
    if hasattr(agent.cfg, 'eps_decay'): agent.cfg.eps_decay = 1.0

    # (C) 同步目标网络
    if hasattr(agent, 'update_target'):
        agent.update_target(hard=True)

    print(f"[Config] 微调保护已开启: LR={new_lr}, Epsilon=0.01")

    # 4. 最佳模型备份
    best_score = 0
    best_path = STUDENT_FINAL_PATH.replace(".torch", "_best.torch")

    # 5. 训练循环
    for run in range(1, num_episodes + 1):
        state, _ = env.reset(seed=run)
        state = np.reshape(state, (1, obs_dim))
        steps = 0

        while True:
            steps += 1
            action = agent.act(state)  # 使用 epsilon=0.01
            next_state_raw, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if done: reward = -1.0  # 简单的失败惩罚

            next_state = np.reshape(next_state_raw, (1, obs_dim))
            agent.step(state, action, reward, next_state, done)
            state = next_state

            if done:
                print(f"微调 Episode {run}/{num_episodes} | Score: {steps}")
                logger.add_score(steps, run)

                # 保存最佳
                if steps >= 475 and steps >= best_score:
                    best_score = steps
                    agent.save(best_path)
                    print(f"  ★ New Best ({steps}) saved.")
                break

    env.close()

    # 6. 恢复最佳模型并保存
    if os.path.exists(best_path):
        print(f"训练结束，恢复最佳模型 ({best_score}分)")
        agent.load(best_path)

    agent.save(STUDENT_FINAL_PATH)
    print(f"[Finish] 最终学生模型保存至: {STUDENT_FINAL_PATH}")

    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Physics Student Runner (Imitation Learning)")

    # 模式选择
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                        help='模式: train(微调训练), eval(评估模型)')

    # 训练参数
    parser.add_argument('--episodes', type=int, default=20,
                        help='微调训练的局数 (默认20，因为有老师基础，不需要练太久)')

    # 评估参数
    parser.add_argument('--path', type=str, default=None,
                        help='指定评估的模型路径 (默认使用 models/student_final.torch)')
    parser.add_argument('--render', action='store_true', help='是否显示画面')

    args = parser.parse_args()

    # 默认的学生模型路径
    default_student_path = "models/student_final.torch"

    # ==========================================
    # 模式 1: 训练模式 (Load Pretrain -> Fine-tune)
    # ==========================================
    if args.mode == 'train':
        print(f"\n=== Starting Student Fine-tuning (Episodes={args.episodes}) ===")

        # 1. 执行微调训练
        student_agent = train_student_agent(num_episodes=args.episodes)

        # 2. 训练完立刻考核 (方便看效果)
        if student_agent:
            print("\n=== Fine-tuning Complete. Immediate Evaluation ===")
            evaluate_agent(
                algorithm="dqn",
                episodes=10,
                render=args.render,
                fps=60,
                if_agent=True,
                agent=student_agent
            )

    # ==========================================
    # 模式 2: 评估模式 (Load Existing Model)
    # ==========================================
    elif args.mode == 'eval':
        # 如果没指定 path，就用默认生成的 student_final.torch
        target_path = args.path if args.path else default_student_path

        if not os.path.exists(target_path):
            print(f" 错误: 找不到文件 {target_path}")
            print("提示: 请先运行 --mode train，或者用 --path 指定正确路径")
        else:
            print(f"\n=== Evaluating Student Model: {target_path} ===")
            evaluate_agent(
                model_path=target_path,
                algorithm="dqn",
                episodes=10,
                render=args.render,
                fps=60
            )