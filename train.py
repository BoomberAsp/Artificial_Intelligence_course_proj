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

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
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
    boundary_distances = [] # 记录边界距离
    mean_score_each_replay = [] # 记录每次训练完后的测试分数

    # Construct agent with default config (students can swap configs here)
    agent = PPOSolver(obs_dim, act_dim, cfg=PPOConfig())
    if config_path:
        agent.load_config(config_path)

    print(f"[Info] Using device: {agent.device}")

    finish = False # 如果中间评估能够达到10次mean为500则停止
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

            # warning
            boundary_warning(next_state_raw,boundary_distances)
            
            # 这里is_renewing在agent.step中触发
            # 记录本集内的所有 "开始训练" 时刻
            if getattr(agent, "is_renewing", False):
                renew_steps_this_episode.append(steps)
                # also evaluate here to see the score
                scores, avg_score = evaluate_agent(algorithm="ppo", 
                                    episodes=10, render=False, if_agent=True, agent=agent)
                mean_score = np.mean(scores)
                mean_score_each_replay.append(mean_score)
                # if mean_score== 500:
                #     finish = True
                #     break
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
        if finish == True:
            break

    env.close()
    
    # 把记录挂在 agent 上，方便外部画图使用
    agent.step_record = step_record              # List[int]
    agent.renew_record = renew_record   
    agent.boundary_distances = boundary_distances
    agent.mean_score_each_replay = mean_score_each_replay

    # Persist the trained model
    if saved:
        agent.save(save_path)
        print(f"[Train] Model saved to {save_path}")
    return agent

# def train_episode_dqn(agent, env) -> DQNSolver:
#     # TODO: 实现单个episode的训练逻辑
#     pass


def plot_training_progress(step_record,
                           renew_record,
                           title: str = "PPO CartPole Training Progress",
                           save_path: str | None = None):
    """
    绘制训练过程中：
      - 每一集的总步数（柱状图）
      - 每一集内部所有触发训练的位置（柱子内的短横线+右侧标记）
    """
    steps = np.array(step_record, dtype=np.int32)
    episodes = np.arange(1, len(steps) + 1)

    assert len(step_record) == len(renew_record), "step_record 与 renew_record 长度必须一致"

    plt.figure(figsize=(14, 7))

    # 1. 柱状图：每集步数
    bar_width = 0.6
    bars = plt.bar(episodes, steps, width=bar_width, alpha=0.6, label="Steps per episode")

    # 2. 在柱子内部画短横线
    for ep_idx, (ep, total_steps) in enumerate(zip(episodes, steps)):
        renew_steps = renew_record[ep_idx]
        for rs in renew_steps:
            if 0 < rs <= total_steps:
                # 短横线（柱子内部）
                line_length = bar_width * 0.5
                x_start = ep - line_length/2
                x_end = ep + line_length/2
                
                plt.plot([x_start, x_end], 
                         [rs, rs],
                         color="red", 
                         linewidth=1.5,
                         alpha=0.7)
                
                # 右侧小圆点（便于识别密集点）
                plt.plot(ep + bar_width/2 + 0.03,
                         rs,
                         marker='.',
                         markersize=4,
                         color='darkred',
                         alpha=0.5)

    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    
    # 图例
    from matplotlib.lines import Line2D
    line_legend = Line2D([0], [0], color="red", linewidth=1.5, alpha=0.7)
    dot_legend = Line2D([0], [0], marker='.', color='w', 
                       markerfacecolor='darkred', markersize=10)
    
    plt.legend(handles=[line_legend, dot_legend], 
               labels=["Training point", "Update location"],
               loc="upper left")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)

    plt.show()

def analyze_boundary_behavior(boundary_distances,
                            title: str = "boundary_behavior",
                           save_path: str | None = None):
    """分析模型在边界附近的行为"""
    # 绘制边界距离分布
    plt.hist(boundary_distances, bins=100)
    plt.xlabel("Distance to Boundary")
    plt.ylabel("Frequency")
    plt.title("Model's Boundary Behavior")
    plt.show()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)

import matplotlib.pyplot as plt
import numpy as np

def evaluation_during_training(mean_score_each_replay,
                               title="Evaluation During Training",
                               save_path="evaluation_during_training.png"):
    """
    绘制训练过程中平均得分的变化曲线
    
    Args:
        mean_score_each_replay: list of float，每个训练周期（replay）的平均得分
        title: 图标题
        save_path: 保存路径，None则不保存
    """
    # 检查输入是否为空
    if not mean_score_each_replay:
        print("Warning: mean_score_each_replay is empty!")
        return
    
    # 转换为numpy数组便于处理
    scores = np.array(mean_score_each_replay)
    episodes = np.arange(1, len(scores) + 1)
    
    # 创建图形
    plt.figure(figsize=(12, 6))
    
    # 绘制折线图
    plt.plot(episodes, scores, 
             marker='o',           # 数据点标记
             markersize=4,         # 标记大小
             linewidth=2,          # 线宽
             alpha=0.7,            # 透明度
             color='royalblue',    # 线条颜色
             label='Mean Score')
    
    # # 可选：添加移动平均线（更平滑的趋势）
    # if len(scores) >= 10:
    #     window = min(10, len(scores) // 5)  # 自适应窗口大小
    #     moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
    #     plt.plot(episodes[window-1:], moving_avg, 
    #              linewidth=2.5, 
    #              color='red', 
    #              linestyle='--', 
    #              alpha=0.8,
    #              label=f'{window}-episode Moving Avg')
    
    # 添加最高分标记
    max_score = np.max(scores)
    max_episode = np.argmax(scores) + 1
    plt.scatter(max_episode, max_score, 
                color='red', 
                s=100,           # 点大小
                zorder=5,        # 显示在最上层
                label=f'Max: {max_score:.1f} (Ep {max_episode})')
    
    # 添加最后得分标记
    final_score = scores[-1]
    plt.scatter(episodes[-1], final_score,
                color='green',
                s=100,
                zorder=5,
                label=f'Final: {final_score:.1f}')
    
    # 设置图表属性
    plt.xlabel('Training Episode / Replay', fontsize=12)
    plt.ylabel('Mean Score', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    
    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 设置y轴从0开始（得分通常是正的）
    plt.ylim(bottom=0, top=max(500, max_score * 1.1))  # CartPole最大500分
    
    # # 如果训练了很多轮，可以设置x轴为对数刻度
    # if len(episodes) > 100:
    #     plt.xscale('log')
    #     plt.xlabel('Training Episode / Replay (log scale)', fontsize=12)
    
    # 添加图例
    plt.legend(loc='best', fontsize=10)
    
    # 添加额外的统计信息文本框
    stats_text = (f'Total Episodes: {len(scores)}\n'
                  f'Average Score: {np.mean(scores):.1f} ± {np.std(scores):.1f}\n'
                  f'Max Score: {max_score:.1f}\n'
                  f'Final Score: {final_score:.1f}')
    
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)
    
    # 如果得分接近CartPole的最大值（500），添加目标线
    if max_score > 400:
        plt.axhline(y=500, color='green', linestyle=':', linewidth=2, alpha=0.5)
        plt.text(episodes[-1], 510, 'Perfect Score (500)', 
                 horizontalalignment='right',
                 color='green', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    # 返回统计数据
    return {
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores)),
        'max': float(max_score),
        'final': float(final_score),
        'episodes': len(scores)
    }



def boundary_warning(state,boundary_distances):
    if state is None or len(state) == 0:
            return
        
    # 提取位置坐标
    if isinstance(state, np.ndarray):
        x_pos = state[0]  # 第一个元素是位置
    elif isinstance(state, list) and len(state) >= 1:
        x_pos = state[0]
    else:
        print(f"Warning: Unexpected state format: {type(state)}")
        return

    # 记录距离边界的距离
    dist = 2.4 - abs(x_pos)
    boundary_distances.append(dist)
    
    # 特别关注边界附近的行为
    if abs(x_pos) > 2.0:
        print(f"危险！位置: {x_pos:.3f}, 到边界距离: {dist:.3f}")

# 临时的修正项：手动强调靠近边界的危险



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
        scores, avg_score = evaluate_agent(algorithm="ppo", episodes=100, render=False, if_agent=True, agent=agent)

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

    # # 设置种子
    # random.seed(89800)
    # np.random.seed(89800)
    # torch.manual_seed(89800)

    # === 模式 1:  PPO ===
    if args.mode == 'ppo':
        print("\n=== Running PPO Training  ===")
        # 先定义好路径，确保“存”和“取”用的是同一个变量
        current_ppo_path = f"models/cartpole_ppo_{TS}.torch"

        # 1. 训练时，显式传入 save_path
        agent = train_ppo(
            num_episodes=2048,
            terminal_penalty=True,
            save_path=current_ppo_path  # <--- 这里必须传，否则它会存成默认名字
        )
        
        plot_training_progress(
            agent.step_record,
            agent.renew_record,
            title="PPO CartPole — Steps and Update Points",
            save_path="ppo_training_steps.png"
        )
        # analyze_boundary_behavior(agent.boundary_distances,
        #                     title = "boundary_behavior",
        #                    save_path="ppo_boundary_behavior.png")
        evaluation_during_training(agent.mean_score_each_replay,
                            title = "evaluation_during_training",
                            save_path="evaluation_during_training.png")
        # 2. 评估时，使用同一个路径
        evaluate_agent(
            model_path=current_ppo_path,
            algorithm="ppo",
            episodes=10,
            render=True,
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
            evaluate_agent(model_path=args.path, algorithm=algo, episodes=100, render=True)
            # python train.py --mode eval --path "models/Best_weight_cartpole_dqn_12m6d18h26min.torch"
            #python train.py --mode eval --path "models/。。。.torch"