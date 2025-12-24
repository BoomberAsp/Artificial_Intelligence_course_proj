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
import json
import os
import random
import time
import numpy as np
import gymnasium as gym
import torch
import time
import matplotlib.pyplot as plt

from agents.cartpole_dqn import DQNSolver, DQNConfig
from agents.cartpole_ac import ACSolver, ACConfig
from agents.cartpole_ppo import PPOSolver, PPOConfig
from agents.cartpole_physics import PhysicsAgent, PhysicsConfig
from agents.cartpole_dqn_priority import PDQNConfig, PDQNSolver, PriorityReplayBuffer, create_pdqn_solver

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
    
    # for evaluation:
    step_record = [] # 记录每次运行的step步数
    renew_record = [] # int,记录后发生renew的步数

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

        # 本集内所有触发训练的 step（可能为空）
        renew_steps_this_episode: list[int] = []
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

            # 这里is_renewing在agent.step中触发
            # 记录本集内的所有 "开始训练" 时刻
            if getattr(agent, "is_renewing", False):
                renew_steps_this_episode.append(steps)
                # 重置标志，避免在下一步又误判
                agent.is_renewing = False
            
            # 强制早停
            if steps>1000:
                done = True
                
            # 7. Episode end: log and break
            if done:
                print(f"Run: {run}, Score: {steps}")
                logger.add_score(steps, run)  # writes CSV + updates score PNG
                renew_record.append(renew_steps_this_episode)
                step_record.append(steps)
                break

    env.close()
    
    # 把记录挂在 agent 上，方便外部画图使用
    agent.step_record = step_record              # List[int]
    agent.renew_record = renew_record            # List[List[int]]
    # Persist the trained model
    if saved:
        agent.save(save_path)
        print(f"[Train] Model saved to {save_path}")
    return agent

# def train_episode_dqn(agent, env) -> DQNSolver:
#     # TODO: 实现单个episode的训练逻辑
#     pass


def train_pdqn(num_episodes: int = 1024, terminal_penalty: bool = True, save_path=MODEL_PATH, saved=True,
               config_path=None) -> PDQNSolver:
    """
    PDQN 训练函数
    基本与 train_dqn 相同，但使用 PDQNSolver
    """
    if saved:
        os.makedirs(MODEL_DIR, exist_ok=True)

    env = gym.make(ENV_NAME)
    logger = ScoreLogger(ENV_NAME)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # 使用 PDQNConfig 和 PDQNSolver
    agent = PDQNSolver(obs_dim, act_dim, cfg=PDQNConfig())
    if config_path:
        agent.load_config(config_path)

    print(f"[Info] Using device: {agent.device}")

    # Episode loop (与 train_dqn 相同)
    for run in range(1, num_episodes + 1):
        state, info = env.reset(seed=run)
        state = np.reshape(state, (1, obs_dim))
        steps = 0

        while True:
            steps += 1
            action = agent.act(state)
            next_state_raw, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if terminal_penalty and done:
                reward = -1.0

            next_state = np.reshape(next_state_raw, (1, obs_dim))
            agent.step(state, action, reward, next_state, done)
            state = next_state

            if done:
                print(f"Run: {run}, Epsilon: {agent.exploration_rate:.3f}, Score: {steps}")
                logger.add_score(steps, run)
                break

    env.close()

    if saved:
        agent.save(save_path)
        print(f"[Train] PDQN Model saved to {save_path}")

    return agent


def plot_training_progress(step_record,
                           renew_record,
                           title: str = "PPO CartPole Training Progress",
                           save_path: str | None = None):
    """
    绘制训练过程中：
      - 每一集的总步数（柱状图）
      - 每一集内部所有触发训练的 step（在柱子上画多条竖线）
    """
    steps = np.array(step_record, dtype=np.int32)
    episodes = np.arange(1, len(steps) + 1)

    # renew_record 是 List[List[int]]，长度应与 step_record 一致
    assert len(step_record) == len(renew_record), "step_record 与 renew_record 长度必须一致"

    plt.figure(figsize=(12, 6))

    # 1. 柱状图：每集步数
    bar_width = 0.8
    plt.bar(episodes, steps, width=bar_width, alpha=0.7, label="Steps per episode")

    # 2. 在每根柱子上画所有更新点
    for ep_idx, (ep, total_steps) in enumerate(zip(episodes, steps)):
        renew_steps = renew_record[ep_idx]  # List[int]
        for rs in renew_steps:
            if 0 < rs <= total_steps:
                # 在该 episode 柱子上画一条红色虚线，表示第 rs 步触发训练
                plt.vlines(x=ep,
                           ymin=0,
                           ymax=rs,
                           colors="red",
                           linestyles="dashed",
                           linewidth=1.0)

    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    # 可以在 legend 里标注线条含义
    from matplotlib.lines import Line2D
    custom_line = Line2D([0], [0], color="red", linestyle="dashed", linewidth=1.0)
    plt.legend(handles=[custom_line], labels=["Training updates"], loc="upper left")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)

    plt.show()

def train_with_config(config, num_episodes=200, save=False) -> tuple[DQNSolver, float, str]:
    env = gym.make("CartPole-v1")
    # ===========根据config创建agent==========
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    agent = None
    description = ""
    save_path = ""

    if isinstance(config, PDQNConfig):
        agent = PDQNSolver(obs_dim, act_dim, cfg=config)
        if save:
            os.makedirs(MODEL_DIR, exist_ok=True)
            print(f"[Info] Using device: {agent.device}")

            params = [
                str("%.4f"%config.gamma),
                str("%.4f"%config.lr),
                str(config.batch_size),
                str(config.memory_size),
                str("%.2f"%config.initial_exploration),
                str("%.2f"%config.eps_start),
                str("%.2f"%config.eps_end),
                str("%.2f"%config.eps_decay),
                str("%.2f"%config.target_update),
                str("%.2f"%config.alpha),  # 添加 PDQN 特定参数
                str("%.2f"%config.beta),  # 添加 PDQN 特定参数
                str("%.2f"%config.beta_increment)  # 添加 PDQN 特定参数
            ]
            description = "_".join(params)
            save_path = MODEL_DIR

    elif isinstance(config, DQNConfig):
        agent = DQNSolver(obs_dim, act_dim, cfg=config)
        if save:
            os.makedirs(MODEL_DIR, exist_ok=True)
            print(f"[Info] Using device: {agent.device}")
            params = [
                str("%.4f"%config.gamma),
                str("%.4f"%config.lr),
                str(config.batch_size),
                str(config.memory_size),
                str("%.2f"%config.initial_exploration),
                str("%.2f"%config.eps_start),
                str("%.2f"%config.eps_end),
                str("%.2f"%config.eps_decay),
                str("%.2f"%config.target_update)
            ]
            description = "_".join(params)  # 加在文件名后用于找寻超参组合
            save_path = MODEL_DIR

    elif isinstance(config, ACConfig):
        agent = ACConfig(obs_dim,act_dim, cfg=config)
        if save:  # 如果要保存模型
            os.makedirs(MODEL_DIR, exist_ok=True)
            print(f"[Info] Using device: {agent.device}")
            params = [
                str("%.2f"%config.gamma),
                str("%.2f"%config.lr),
                str("%.2f"%config.batch_size),
                str("%.2f"%config.memory_size),
                str("%.2f"%config.value_coef),
                str("%.2f"%config.entropy_coef)
            ]
            description = "_".join(params)  # 加在文件名后用于找寻超参组合
            save_path = MODEL_DIR

    elif isinstance(config, PPOConfig):
        agent = PPOSolver(obs_dim,act_dim, cfg=config)
        if save:  # 如果要保存模型
            os.makedirs(MODEL_DIR, exist_ok=True)
            print(f"[Info] Using device: {agent.device}")
            params = [
                str("%.2f"%config.gamma),
                str("%.2f"%config.learning_rate),
                str("%.2f"%config.memory_size),
                str("%.2f"%config.value_coef),
                str("%.2f"%config.entropy_coef),
                str("%.2f"%config.clip_eps),
                str("%.2f"%config.lambda_gae),
                str("%.2f"%config.minibatch_size),
                str("%.2f"%config.epoch),
            ]
            description = "_".join(params)

    elif isinstance(config, PhysicsConfig):
        agent = PhysicsAgent(obs_dim, act_dim, cfg=config)
        if save:  # 如果要保存模型
            os.makedirs(MODEL_DIR, exist_ok=True)
            # PhysicsAgent通常运行在CPU上，如果你的类里没有定义.device，可以将下面这行注释掉
            # print(f"[Info] Using device: {agent.device}")
            params = [
                str("%.2f"%config.theta_coef),
                str("%.2f"%config.omega_coef),
                str("%.2f"%config.pos_coef),
                str("%.2f"%config.vel_coef)
            ]



            description = "_".join(params)  # 加在文件名后用于找寻超参组合
            save_path = MODEL_DIR



    # TODO:实现其它agent时需要添加，用于调取对应的agent
    

    # =======================================


    

    # ==========训练、评估逻辑==========
    avg_score = 0.0
    if isinstance(agent, PDQNSolver):
        saved_path = os.path.join(save_path, f"cartpole_pdqn_{description}.torch")
        agent = train_pdqn(num_episodes=num_episodes,

                           saved=save, save_path=saved_path)

        scores, avg_score = evaluate_agent(algorithm="pdqn", episodes=100, render=False, if_agent=True, agent=agent)
    elif isinstance(agent, DQNSolver):
        saved_path = os.path.join(save_path, f"cartpole_dqn_{description}.torch")
        agent = train_dqn(num_episodes=num_episodes,
                          saved=save, save_path=saved_path)
        scores, avg_score = evaluate_agent(algorithm="dqn", episodes=100, render=False, if_agent=True, agent=agent)
    elif isinstance(agent, PPOSolver):
        saved_path = os.path.join(save_path, f"cartpole_ppo_{description}.torch")
        agent = train_ppo(num_episodes=num_episodes,
                          saved=save, save_path=saved_path)
        scores, avg_score = evaluate_agent(algorithm="ac", episodes=100, render=False, if_agent=True, agent=agent)




    # TODO:实现其它agent时需要添加，用于调取对应的训练函数与评估函数
    # ================================
    if not agent:
        raise ValueError("Training failed, agent is None")
    elif avg_score == 0.0:
        raise ValueError("Evaluation failed, avg_score is 0.0")

    return agent, avg_score, saved_path


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
        elif algorithm.lower() == "pdqn":
            agent = PDQNSolver(obs_dim, act_dim, cfg=PDQNConfig())
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



def main():
    parser = argparse.ArgumentParser(description="CartPole Agent Runner")
    parser.add_argument('--mode', type=str, default='ppo', choices=['dqn', 'pdqn', 'ppo', 'physics', 'eval'],
                        help='选择运行模式: dqn, pdqn, ppo, physics, eval(评估)')
    parser.add_argument('--path', type=str, default=None, help='模型路径 (仅eval模式)')
    parser.add_argument('--render', action='store_true', help='是否渲染画面')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径 (JSON格式)')  # 新增参数
    parser.add_argument('--episodes', type=int, default=128, help='训练回合数')  # 新增参数

    args = parser.parse_args()

    # 设置种子
    random.seed(89800)
    np.random.seed(89800)
    torch.manual_seed(89800)

    # 创建配置函数
    def load_config_from_file(config_path: str, algorithm: str):
        """从JSON文件加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)

            # 根据算法类型创建对应的配置对象
            if algorithm == 'dqn':
                # 确保所有必要的参数都存在
                default_config = DQNConfig()
                for key, value in config_dict.items():
                    if hasattr(default_config, key):
                        setattr(default_config, key, value)
                return default_config
            elif algorithm == 'ppo':
                default_config = PPOConfig()
                for key, value in config_dict.items():
                    if hasattr(default_config, key):
                        setattr(default_config, key, value)
                return default_config
            elif algorithm == 'pdqn':
                default_config = PDQNConfig()
                for key, value in config_dict.items():
                    if hasattr(default_config, key):
                        setattr(default_config, key, value)
                return default_config
            elif algorithm == 'physics':
                default_config = PhysicsConfig()
                for key, value in config_dict.items():
                    if hasattr(default_config, key):
                        setattr(default_config, key, value)
                return default_config
            else:
                print(f"警告: 未知算法 {algorithm}，使用默认配置")
                return None

        except FileNotFoundError:
            print(f"错误: 配置文件 {config_path} 不存在")
            return None
        except json.JSONDecodeError:
            print(f"错误: 配置文件 {config_path} 格式错误")
            return None
        except Exception as e:
            print(f"错误: 加载配置文件失败: {e}")
            return None

    # === 模式 1:  PPO ===
    if args.mode == 'ppo':
        print("\n=== Running PPO Training ===")

        # 优先使用配置文件，如果没有则使用默认配置
        if args.config:
            config = load_config_from_file(args.config, 'ppo')
            if config is None:
                print("使用默认配置继续训练...")
                config = PPOConfig()
        else:
            config = PPOConfig()

        print(f"使用配置: {config.__dict__}")

        # 创建agent
        obs_dim = 4  # CartPole状态维度
        act_dim = 2  # CartPole动作维度
        agent = PPOSolver(obs_dim, act_dim, cfg=config)

        # 训练
        print(f"开始训练，总回合数: {args.episodes}")
        # agent = train_ppo(
        #     num_episodes=args.episodes,
        #     terminal_penalty=True,
        #     save_path=f"models/cartpole_ppo_{TS}.torch",
        #     saved=True,
        #     config_path=args.config  # 不再需要config_path，因为配置已通过cfg传入
        # )
        agent,avg_score,saved_path = train_with_config(config=config,
                                  num_episodes=args.episodes,
                                  save=True)

        print(f"训练完成，平均得分: {avg_score:.2f}")

        # 绘制训练进度
        if hasattr(agent, 'step_record') and hasattr(agent, 'renew_record'):
            plot_training_progress(
                agent.step_record,
                agent.renew_record,
                title="PPO CartPole — Steps and Update Points",
                save_path=f"ppo_training_steps_{TS}.png"
            )

        # 评估
        evaluate_agent(
            model_path=saved_path,
            algorithm="ppo",
            episodes=10,
            render=args.render,
            fps=60
        )

    # === 模式 2:  DQN ===
    elif args.mode == 'dqn':
        print("\n=== Running DQN Training ===")

        # 优先使用配置文件
        if args.config:
            config = load_config_from_file(args.config, 'dqn')
            if config is None:
                print("使用默认配置继续训练...")
                config = DQNConfig()
        else:
            config = DQNConfig()

        print(f"使用配置: {config.__dict__}")

        # 使用train_with_config进行训练
        agent, avg_score, saved_path = train_with_config(
            config=config,
            num_episodes=args.episodes,
            save=True
        )

        print(f"训练完成，平均得分: {avg_score:.2f}")

        # 评估
        # 需要从配置参数生成模型文件名
        # params = [
        #     str(config.gamma),
        #     str(config.lr),
        #     str(config.batch_size),
        #     str(config.memory_size),
        #     str(config.initial_exploration),
        #     str(config.eps_start),
        #     str(config.eps_end),
        #     str(config.eps_decay),
        #     str(config.target_update)
        # ]
        # description = "_".join(params)
        # model_filename = f"cartpole_dqn_{description}.torch"

        evaluate_agent(
            model_path=saved_path,
            algorithm="dqn",
            episodes=10,
            render=args.render,
            fps=60
        )

    # === 模式 3:  PDQN ===
    elif args.mode == 'pdqn':
        print("\n=== Running PDQN Training ===")

        # 优先使用配置文件
        if args.config:
            config = load_config_from_file(args.config, 'pdqn')
            if config is None:
                print("使用默认配置继续训练...")
                config = PDQNConfig()
        else:
            config = PDQNConfig()

        print(f"使用配置: {config.__dict__}")

        # 使用train_with_config进行训练
        agent, avg_score, saved_path = train_with_config(
            config=config,
            num_episodes=args.episodes,
            save=True
        )

        print(f"训练完成，平均得分: {avg_score:.2f}")

        # 评估
        # params = [
        #     str(config.gamma),
        #     str(config.lr),
        #     str(config.batch_size),
        #     str(config.memory_size),
        #     str(config.initial_exploration),
        #     str(config.eps_start),
        #     str(config.eps_end),
        #     str(config.eps_decay),
        #     str(config.target_update),
        #     str(config.alpha),
        #     str(config.beta),
        #     str(config.beta_increment)
        # ]
        # description = "_".join(params)
        # model_filename = f"cartpole_pdqn_{description}.torch"

        evaluate_agent(
            model_path=saved_path,
            algorithm="pdqn",
            episodes=10,
            render=args.render,
            fps=60
        )

    # === 模式 4:  Physics (Teacher) ===
    elif args.mode == 'physics':
        print("\n=== Running Physics Teacher Demo ===")

        # 优先使用配置文件
        if args.config:
            config = load_config_from_file(args.config, 'physics')
            if config is None:
                print("使用默认配置继续训练...")
                config = PhysicsConfig(theta_coef=1.0, omega_coef=1.0, pos_coef=0.1, vel_coef=0.1)
        else:
            config = PhysicsConfig(theta_coef=1.0, omega_coef=1.0, pos_coef=0.1, vel_coef=0.1)

        teacher = PhysicsAgent(4, 2, cfg=config)
        evaluate_agent(algorithm="physics",
                       episodes=5,
                       render=args.render,
                       fps=60,
                       if_agent=True,
                       agent=teacher)

    # === 模式 5: 通用评估 ===
    elif args.mode == 'eval':
        if not args.path:
            print("请提供模型路径: --path models/xxx.torch")
        else:
            # 自动检测算法类型
            if "dqn" in args.path.lower():
                algo = "dqn"
            elif "ppo" in args.path.lower():
                algo = "ppo"
            elif "pdqn" in args.path.lower():
                algo = "pdqn"
            else:
                algo = "dqn"  # 默认

            evaluate_agent(
                model_path=args.path,
                algorithm=algo,
                episodes=100,
                render=args.render,
                fps=60
            )


if __name__ == "__main__":
    # [新增] 命令行参数控制
    # parser = argparse.ArgumentParser(description="CartPole Agent Runner")
    # parser.add_argument('--mode', type=str, default='ppo', choices=['dqn', 'pdqn', 'ppo', 'physics', 'eval'],
    #                     help='选择运行模式: dqn, pdqn, ppo, physics, eval(评估)')
    # parser.add_argument('--path', type=str, default=None, help='模型路径 (仅eval模式)')
    # parser.add_argument('--render', action='store_true', help='是否渲染画面')
    #
    # args = parser.parse_args()
    #
    # # 设置种子
    # random.seed(89800)
    # np.random.seed(89800)
    # torch.manual_seed(89800)
    #
    # # === 模式 1:  PPO ===
    # if args.mode == 'ppo':
    #     print("\n=== Running PPO Training  ===")
    #     # 先定义好路径，确保“存”和“取”用的是同一个变量
    #     current_ppo_path = f"models/cartpole_ppo_{TS}.torch"
    #
    #     # 1. 训练时，显式传入 save_path
    #     agent = train_ppo(
    #         num_episodes=128,
    #         terminal_penalty=True,
    #         save_path=current_ppo_path  # <--- 这里必须传，否则它会存成默认名字
    #     )
    #
    #     plot_training_progress(
    #         agent.step_record,
    #         agent.renew_record,
    #         title="PPO CartPole — Steps and Update Points",
    #         save_path="ppo_training_steps.png"
    #     )
    #     # 2. 评估时，使用同一个路径
    #     evaluate_agent(
    #         model_path=current_ppo_path,
    #         algorithm="ppo",
    #         episodes=10,
    #         render=False,
    #         fps=60
    #     )
    # # === 模式 2:  DQN (Student) ===
    # elif args.mode == 'dqn':
    #     print("\n=== Standard DQN Training (From Scratch) ===")
    #     # 直接 python train.py --mode dqn
    #     agent = train_dqn(num_episodes=300, terminal_penalty=True)
    #     evaluate_agent(model_path=f"models/cartpole_dqn_{TS}.torch", algorithm="dqn", episodes=10, render=False, fps=60)
    #
    #     # 添加 PDQN 模式
    # elif args.mode == 'pdqn':
    #     print("\n=== Running PDQN Training (Priority DQN) ===")
    #     current_pdqn_path = f"models/cartpole_pdqn_{TS}.torch"
    #
    #     # 注意：这里需要创建一个 PDQNConfig 配置
    #     config = PDQNConfig()
    #     # 可以调整 PDQN 特有参数
    #     config.alpha = 0.6  # 优先级强度
    #     config.beta = 0.4  # 重要性采样初始值
    #     config.beta_increment = 0.001
    #
    #     agent = train_with_config(
    #         config=config,
    #         num_episodes=300,
    #         save=True
    #     )
    #
    #     evaluate_agent(
    #         model_path=current_pdqn_path,
    #         algorithm="pdqn",
    #         episodes=10,
    #         render=False,
    #         fps=60
    #     )
    #
    #
    #
    # # === 模式 3:  Physics (Teacher) ===
    # elif args.mode == 'physics':
    #     print("\n=== Running Physics Teacher Demo ===")
    #     cfg = PhysicsConfig(theta_coef=1.0, omega_coef=1.0, pos_coef=0.1, vel_coef=0.1)
    #     teacher = PhysicsAgent(4, 2, cfg=cfg)
    #     evaluate_agent(algorithm="physics", episodes=5, render=False, fps=0, if_agent=True, agent=teacher)
    #
    # # === 模式 4: 通用评估 ===
    # elif args.mode == 'eval':
    #     if not args.path:
    #         print("请提供模型路径: --path models/xxx.torch")
    #     else:
    #         algo = "dqn" if "dqn" in args.path else "ppo"
    #         evaluate_agent(model_path=args.path, algorithm=algo, episodes=100, render=False)


    main()
            # python train.py --mode eval --path "models/Best_weight_cartpole_dqn_12m6d18h26min.torch"
            #python train.py --mode eval --path "models/。。。.torch"