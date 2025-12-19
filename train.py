""""
CartPole Training & Evaluation (PyTorch + Gymnasium)
---------------------------------------------------
- Trains a DQN agent and logs scores via ScoreLogger (PNG + CSV)
- Saves model to ./models/cartpole_dqn.torch
- Evaluates from a saved model (render optional)

Student reading map:
  1) train(): env loop → agent.act() → env.step() → agent.step() [Encapsulated]
  2) evaluate(): loads saved model and runs agent.act(evaluation_mode=True)
"""

from __future__ import annotations

import argparse
import os
import random
import time
import numpy as np
import gymnasium as gym
import torch
import time

from agents.cartpole_dqn import DQNSolver, DQNConfig
from agents.cartpole_ac import ACSolver, ACConfig
from agents.cartpole_ppo import PPOSolver, PPOConfig
from agents.cartpole_physics import PhysicsAgent, PhysicsConfig

from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"
MODEL_DIR = "models"
TS = (str(time.localtime().tm_mon)+"m"
      + str(time.localtime().tm_mday)+"d"
      + str(time.localtime().tm_hour)+"h"
      + str(time.localtime().tm_min)+"min")

MODEL_PATH = os.path.join(MODEL_DIR, f"cartpole_dqn_{TS}.torch")


def train_dqn(num_episodes: int = 1024, terminal_penalty: bool = True, save_path = MODEL_PATH, saved = True, config_path = None) -> DQNSolver:
    """
    Main training loop:
      - Creates the environment and agent
      - For each episode:
          * Reset env → get initial state
          * Loop: select action, step environment, call agent.step()
          * Log episode score with ScoreLogger
      - Saves the trained model to disk
    """

    if saved:
        os.makedirs(MODEL_DIR, exist_ok=True)

    # Create CartPole environment (no render during training for speed)
    env = gym.make(ENV_NAME)
    logger = ScoreLogger(ENV_NAME)

    # Infer observation/action dimensions from the env spaces
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Construct agent with default config (students can swap configs here)
    agent = DQNSolver(obs_dim, act_dim, cfg=DQNConfig())
    if config_path:
        agent.load_config(config_path)

    print(f"[Info] Using device: {agent.device}")

    # Episode loop
    for run in range(1, num_episodes + 1):
        # Gymnasium reset returns (obs, info). Seed for repeatability.
        state, info = env.reset(seed=run)
        state = np.reshape(state, (1, obs_dim))
        steps = 0

        while True:
            steps += 1

            # 1. ε-greedy action from the agent (training mode)
            #    state shape is [1, obs_dim]
            action = agent.act(state)

            # 2. Gymnasium step returns: obs', reward, terminated, truncated, info
            next_state_raw, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 3. Optional small terminal penalty (encourage agent to avoid failure)
            if terminal_penalty and done:
                reward = -1.0

            # 4. Reshape next_state for agent and next loop iteration
            next_state = np.reshape(next_state_raw, (1, obs_dim))

            # 5. Give (s, a, r, s', done) to the agent, which handles
            #    remembering and learning internally.
            agent.step(state, action, reward, next_state, done)

            # 6. Move to next state
            state = next_state

            # 7. Episode end: log and break
            if done:
                print(f"Run: {run}, Epsilon: {agent.exploration_rate:.3f}, Score: {steps}")
                logger.add_score(steps, run)  # writes CSV + updates score PNG
                break

    env.close()
    # Persist the trained model
    if saved:
        agent.save(save_path)
        print(f"[Train] Model saved to {save_path}")
    return agent
def train_ppo(num_episodes: int = 200, terminal_penalty: bool = True, save_path = MODEL_PATH, saved = True, config_path = None) -> PPOSolver:
    """
    Main training loop:
      - Creates the environment and agent
      - For each episode:
          * Reset env → get initial state
          * Loop: select action, step environment, call agent.step()
          * Log episode score with ScoreLogger
      - Saves the trained model to disk
    """
    print("start ppo training")
    if saved:
        os.makedirs(MODEL_DIR, exist_ok=True)

    # Create CartPole environment (no render during training for speed)
    env = gym.make(ENV_NAME)
    logger = ScoreLogger(ENV_NAME)

    # Infer observation/action dimensions from the env spaces
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Construct agent with default config (students can swap configs here)
    agent = PPOSolver(obs_dim, act_dim, cfg=PPOConfig())
    if config_path:
        agent.load_config(config_path)

    print(f"[Info] Using device: {agent.device}")

    # Episode loop
    for run in range(1, num_episodes + 1):
        # Gymnasium reset returns (obs, info). Seed for repeatability.
        state, info = env.reset(seed=run)
        state = np.reshape(state, (1, obs_dim))
        steps = 0

        while True:
            steps += 1

            # 1. ε-greedy action from the agent (training mode)
            #    state shape is [1, obs_dim]
            action = agent.act(state)

            # 2. Gymnasium step returns: obs', reward, terminated, truncated, info
            next_state_raw, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 3. Optional small terminal penalty (encourage agent to avoid failure)
            if terminal_penalty and done:
                reward = -1.0
            
            # 4. Reshape next_state for agent and next loop iteration
            next_state = np.reshape(next_state_raw, (1, obs_dim))

            # 5. Give (s, a, r, done, logp?) to the agent, which handles
            #    remembering and learning internally.
            agent.step(state, action, reward, done)

            # 6. Move to next state
            state = next_state

            # 7. Episode end: log and break
            if done:
                print(f"Run: {run}, Score: {steps}")
                logger.add_score(steps, run)  # writes CSV + updates score PNG
                break

    env.close()
    # Persist the trained model
    if saved:
        agent.save(save_path)
        print(f"[Train] Model saved to {save_path}")
    return agent

# def train_episode_dqn(agent, env) -> DQNSolver:
#     # TODO: 实现单个episode的训练逻辑
#     pass


def train_with_config(config, num_episodes=200, save=False) -> tuple[DQNSolver, float]:
    env = gym.make("CartPole-v1")
    # ===========根据config创建agent==========
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    agent = None
    description = ""
    save_path = ""
    if isinstance(config, DQNConfig):
        agent = DQNSolver(obs_dim, act_dim, cfg=config)
        if save:
            os.makedirs(MODEL_DIR, exist_ok=True)
            print(f"[Info] Using device: {agent.device}")
            params = [
                str(config.gamma),
                str(config.lr),
                str(config.batch_size),
                str(config.memory_size),
                str(config.initial_exploration),
                str(config.eps_start),
                str(config.eps_end),
                str(config.eps_decay),
                str(config.target_update)
            ]
            description = "_".join(params)  # 加在文件名后用于找寻超参组合
            save_path = MODEL_DIR
    elif isinstance(config, ACConfig):
        agent = ACConfig(obs_dim,act_dim, cfg=config)
        if save:  # 如果要保存模型
            os.makedirs(MODEL_DIR, exist_ok=True)
            print(f"[Info] Using device: {agent.device}")
            params = [
                str(config.gamma),
                str(config.lr),
                str(config.batch_size),
                str(config.memory_size),
                str(config.value_coef),
                str(config.entropy_coef)
            ]
            description = "_".join(params)  # 加在文件名后用于找寻超参组合
            save_path = MODEL_DIR

    elif isinstance(config, PPOConfig):
        agent = PPOSolver(obs_dim,act_dim, cfg=config)
        if save:  # 如果要保存模型
            os.makedirs(MODEL_DIR, exist_ok=True)
            print(f"[Info] Using device: {agent.device}")
            params = [
                str(config.gamma),
                str(config.learning_rate),
                str(config.memory_size),
                str(config.value_coef),
                str(config.entropy_coef),
                str(config.clip_eps),
                str(config.lambda_gae),
                str(config.minibatch_size),
                str(config.epoch),
            ]

    elif isinstance(config, PhysicsConfig):
        agent = PhysicsAgent(obs_dim, act_dim, cfg=config)
        if save:  # 如果要保存模型
            os.makedirs(MODEL_DIR, exist_ok=True)
            # PhysicsAgent通常运行在CPU上，如果你的类里没有定义.device，可以将下面这行注释掉
            # print(f"[Info] Using device: {agent.device}")
            params = [
                str(config.theta_coef),
                str(config.omega_coef),
                str(config.pos_coef),
                str(config.vel_coef)
            ]



            description = "_".join(params)  # 加在文件名后用于找寻超参组合
            save_path = MODEL_DIR
    # TODO:实现其它agent时需要添加，用于调取对应的agent
    

    # =======================================


    

    # ==========训练、评估逻辑==========
    avg_score = 0.0
    if isinstance(agent, DQNSolver):
        agent = train_dqn(num_episodes=num_episodes,
                          saved=save, save_path=os.path.join(save_path, f"cartpole_dqn_{description}.torch"))
        scores, avg_score = evaluate_agent(algorithm="dqn", episodes=100, render=False, if_agent=True, agent=agent)
    elif isinstance(agent, PPOSolver):
        agent = train_ppo(num_episodes=num_episodes,
                          saved=save, save_path=os.path.join(save_path, f"cartpole_ppo_{description}.torch"))
        scores, avg_score = evaluate_agent(algorithm="ac", episodes=100, render=False, if_agent=True, agent=agent)

    # TODO:实现其它agent时需要添加，用于调取对应的训练函数与评估函数
    # ================================
    if not agent:
        raise ValueError("Training failed, agent is None")
    elif avg_score == 0.0:
        raise ValueError("Evaluation failed, avg_score is 0.0")

    return agent, avg_score


def evaluate_agent(model_path: str | None = None,
                 algorithm: str = "dqn",
                 episodes: int = 5,
                 render: bool = True,
                 fps: int = 60,
                 if_agent=False,
                 agent=None) -> tuple[list[int], float]:
    """
    Evaluate a trained agent in the environment using greedy policy (no ε).
    - Loads weights from disk
    - Optionally renders (pygame window)
    - Reports per-episode steps and average

    Args:
        model_path: If None, auto-pick the first .torch file under ./models
        algorithm: Reserved hook if you later support PPO/A2C agents
        episodes: Number of evaluation episodes
        render: Whether to show a window; set False for headless CI
        fps: Target frame-rate during render (sleep-based pacing)
        if_agent: 是否传入agent
        agent: 传入的agent实例
    Returns:
        A tuple of (list of episode steps, average steps)
        e.g., ([200, 195, 210], 201.67)
        where higher is better (max 500 for CartPole-v1) and step is equivalent to score.
    """
    # Resolve model path
    model_dir = MODEL_DIR
    if not if_agent:  # 如果不用传入agent
        if model_path is None:
            candidates = [f for f in os.listdir(model_dir) if f.endswith(".torch")]
            if not candidates:
                raise FileNotFoundError(f"No saved model found in '{model_dir}/'. Please train first.")
            model_path = os.path.join(model_dir, candidates[0])
            print(f"[Eval] Using detected model: {model_path}")
        else:
            print(f"[Eval] Using provided model: {model_path}")

        # Create env for evaluation; 'human' enables pygame-based rendering
        render_mode = "human" if render else None
        
        tmp_env = gym.make(ENV_NAME)
        obs_dim = tmp_env.observation_space.shape[0]
        act_dim = tmp_env.action_space.n
        tmp_env.close()


        # TODO: (If you add PPO/A2C later, pick their agent classes by 'algorithm' here.)
        if algorithm.lower() == "dqn":
            agent = DQNSolver(obs_dim, act_dim, cfg=DQNConfig())
        elif algorithm.lower() == "ppo":
            agent = PPOSolver(obs_dim, act_dim, cfg=PPOConfig())
        elif algorithm.lower() == "physics":
            agent = PhysicsAgent(obs_dim, act_dim, cfg=PhysicsConfig())
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

            # Load trained weights (Physics 不需要加载模型)
        if algorithm.lower() != "physics":
            agent.load(model_path)
            print(f"[Eval] Loaded {algorithm.upper()} model from: {model_path}")

        else:  # 如果要传入agent
            if agent:  # 检查是否传入了agent
                pass
            else:  # 如果没有传入agent，报错
                raise ValueError("evaluate_agent called with if_agent=True but agent is None!")

    scores = []
    # Sleep interval to approximate fps; set 0 for fastest evaluation
    dt = (1.0 / fps) if render and fps else 0.0

    # 2) 这里统一创建正式用来评估的 env
    render_mode = "human" if render else None
    env = gym.make(ENV_NAME, render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]
    
    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=10_000 + ep)
        state = np.reshape(state, (1, obs_dim))
        done = False
        steps = 0

        while not done:
            # Greedy action (no exploration) by calling act() in evaluation mode
            if algorithm.lower() == "physics":
                action = agent.act(state)
            else:
                try:
                    action = agent.act(state, evaluation_mode=True)
                except TypeError:
                    action = agent.act(state)  # Fallback

            # Step env forward
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = np.reshape(next_state, (1, obs_dim))
            steps += 1

            # Slow down rendering to be watchable
            if dt > 0:
                time.sleep(dt)

        scores.append(steps)
        print(f"[Eval] Episode {ep}: steps={steps}")

    env.close()
    avg = float(np.mean(scores)) if scores else 0.0
    print(f"[Eval] Average over {episodes} episodes: {avg:.2f}")
    return scores, avg


'''
if __name__ == "__main__":
    # Example: quick training then a short evaluation
    random.seed(89800)
    np.random.seed(89800)
    torch.manual_seed(89800)
    # 
    agent = train_ppo(num_episodes=128, terminal_penalty=True)
    evaluate_agent(model_path=f"models/cartpole_ppo_{TS}.torch", algorithm="ppo", episodes=500, render=True, fps=60)
'''

if __name__ == "__main__":
    # [新增] 命令行参数控制
    parser = argparse.ArgumentParser(description="CartPole Agent Runner")
    parser.add_argument('--mode', type=str, default='ppo', choices=['dqn', 'ppo', 'physics', 'eval'],
                        help='选择运行模式: dqn(你的学生), ppo(队友的代码), physics(老师演示), eval(评估)')
    parser.add_argument('--path', type=str, default=None, help='模型路径 (仅eval模式)')
    parser.add_argument('--render', action='store_true', help='是否渲染画面')

    args = parser.parse_args()

    # 设置种子
    random.seed(89800)
    np.random.seed(89800)
    torch.manual_seed(89800)

    # === 模式 1:  PPO ===
    if args.mode == 'ppo':
        print("\n=== Running PPO Training  ===")
        # 先定义好路径，确保“存”和“取”用的是同一个变量
        current_ppo_path = f"models/cartpole_ppo_{TS}.torch"

        # 1. 训练时，显式传入 save_path
        agent = train_ppo(
            num_episodes=128,
            terminal_penalty=True,
            save_path=current_ppo_path  # <--- 这里必须传，否则它会存成默认名字
        )

        # 2. 评估时，使用同一个路径
        evaluate_agent(
            model_path=current_ppo_path,
            algorithm="ppo",
            episodes=10,
            render=False,
            fps=60
        )
    # === 模式 2:  DQN (Student) ===
    elif args.mode == 'dqn':
        print("\n=== Standard DQN Training (From Scratch) ===")
        # 直接 python train.py --mode dqn
        agent = train_dqn(num_episodes=300, terminal_penalty=True)
        evaluate_agent(model_path=f"models/cartpole_dqn_{TS}.torch", algorithm="dqn", episodes=10, render=False, fps=60)



    # === 模式 3:  Physics (Teacher) ===
    elif args.mode == 'physics':
        print("\n=== Running Physics Teacher Demo ===")
        cfg = PhysicsConfig(theta_coef=1.0, omega_coef=1.0, pos_coef=0.1, vel_coef=0.1)
        teacher = PhysicsAgent(4, 2, cfg=cfg)
        evaluate_agent(algorithm="physics", episodes=5, render=False, fps=0, if_agent=True, agent=teacher)

    # === 模式 4: 通用评估 ===
    elif args.mode == 'eval':
        if not args.path:
            print("请提供模型路径: --path models/xxx.torch")
        else:
            algo = "dqn" if "dqn" in args.path else "ppo"
            evaluate_agent(model_path=args.path, algorithm=algo, episodes=100, render=False)
            # python train.py --mode eval --path "models/Best_weight_cartpole_dqn_12m6d18h26min.torch"
            #python train.py --mode eval --path "models/。。。.torch"