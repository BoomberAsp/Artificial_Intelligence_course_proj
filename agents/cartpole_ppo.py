"""
PyTorch  for CartPole (Gymnasium)
------------------------------------


"""

from __future__ import annotations
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import json
from pathlib import Path


# -----------------------------
# PPO-specific Hyperparameters
# -----------------------------
GAMMA = 0.99             # reward discount in TD error
VALUE_COEF = 0.54        # critic loss 权重
ENTROPY_COEF = 0.002     # 熵正则，鼓励探索
LR = 0.00015              # learning rate for critic
LAMBDA_GAE = 0.95       # GAE的指数加权平均，用来平衡：0等价TD(0)，1等价Monte Carlo
CLIP_EPS = 0.2        # clip参数，用来限制跨度太大的参数变化
# --------取样训练-------
MEMORY_SIZE = 32       # 单次获得的数据批量
MINIBATCH_SIZE = 64     # 每次从batch里面找多长的sub序列
EPOCH = 16              # 从堆里面取多少次



class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        # policy head
        self.policy_head = nn.Linear(128, act_dim)
        # value head
        self.value_head = nn.Linear(128,1)
        
    def forward(self, x):
        x = self.shared(x)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        dist = Categorical(logits = logits)
        return dist, value


class ReplayBuffer:
    """
    FIFO replay buffer storing transitions as numpy arrays.
    - We convert to torch.Tensor only when sampling a batch.
    - Each entry stores (s, a, r, s', mask, logp_old, value) where mask = 0 if terminal else 1.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buf: Deque[Tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=capacity)
    # TODO[PPO]: 对 PPO 来说，这里更像是一个 "RolloutBuffer":
    #   - 不需要很大的 capacity 和随机采样，通常容量=cfg.rollout_steps
    #   - 每一轮只存当前策略下采样到的那一批 (s, a, r, done, logp_old, value)
    #   - 更新完 PPO 之后要 clear()，然后重新收集下一轮
    #   - 也就是说：
    #       push(...) 不变，但需要增加 logp_old, value 两个字段
    #       sample(batch_size) 在 PPO 中会改成：
    #           - 要么直接返回整批数据（不打乱）
    #           - 要么配合 update_epochs/minibatch_size 手动打乱 indices 再切 mini-batch
    def push(self, s, a, r, done, logp, value):
        s = np.asarray(s)

        if s.ndim == 2 and s.shape[0] == 1:
            s = s.squeeze(0)

        self.buf.append((s, a, r, 0.0 if done else 1.0, logp, value))

    def get_all(self, batch_size: int):
        #   - 典型做法：一次把整个 buffer 拿出来，然后自己在 update_ppo() 里打乱 indices 切 mini-batch
        # 我先就一个batch来处理
        batch = self.buf
        s, a, r, m, logp_old, value = zip(*batch)
        # Shapes after stacking:
        #  s, s2: [B, obs_dim], a: [B], r: [B], m: [B]
        return (
            np.stack(s, axis=0),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(m, dtype=np.float32),
            np.array(logp_old, dtype=np.float32),
            np.array(value, dtype=np.float32)
        )
        
    

    def clear(self):
        self.buf.clear()
        return
    
    def __len__(self):
        return len(self.buf)


@dataclass
class PPOConfig:
    """
    # class PPOConfig:
    #   - gamma, lr, rollout_steps, update_epochs, minibatch_size
    #   - clip_eps, value_coef, entropy_coef, lambda_gae
    #   - device 同样保留

    """
    lr: float = LR
    gamma : float = GAMMA                    # reward discount in TD error
    value_coef : float = VALUE_COEF        # critic loss 权重
    entropy_coef : float = ENTROPY_COEF    # 熵正则，鼓励探索
    clip_eps : float = CLIP_EPS
    lambda_gae : float = LAMBDA_GAE
    memory_size : int = MEMORY_SIZE # 多少批训练一次
    minibatch_size: int = MINIBATCH_SIZE # 每次从数据中取出的轨迹长度
    epoch : int = EPOCH # 每次取多少条
    # max_grad_norm: float = MAX_GRAD_NORM
    
    # Auto-select CUDA if available; CPU is perfectly fine for CartPole
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PPOSolver:
    """
    the following explanation:
    - self.net: ActorCritic 网络，输出 (logits, V(s))
    - self.optim: Adam 优化器，优化 actor 和 critic 所有参数
    - self.memory: ReplayBuffer，用来存当前这一轮 rollout 的数据
    - self.steps: 总交互步数计数
    """

    def __init__(self, observation_space: int, action_space: int, cfg: PPOConfig | None = None):
        # Store dimensions and hyperparameters
        self.obs_dim = observation_space
        self.act_dim = action_space
        self.cfg = cfg or PPOConfig()

        # Choose device (GPU if available, else CPU)
        self.device = torch.device(self.cfg.device)

        # Generate ActorCritic network
        self.net = ActorCritic(self.obs_dim, self.act_dim).to(self.device)

        # Optimizer over online network parameters
        self.optim = optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        
        # Experience replay memory        
        self.memory = ReplayBuffer(self.cfg.memory_size)
        
        self._last_state = None
        self._last_action = None
        self._last_logp = None
        self._last_value = None
        
        self.steps = 0
        self.update_idx = 0  # 第几轮 PPO 更新

        # Global counters, 训练过程中的简单日志，方便画图
        self.train_log = {
            "update_idx": [],
            "mean_return": [],
            "lr": [],
            "epoch": [],
            "memory_size": [],
            "minibatch_size": [],
            "approx_kl": [],
            "clip_fraction": [],
        }

        # 一些自适应的上下界，可以放在 cfg 里，也可以直接写死
        self.cfg.lr_min = getattr(self.cfg, "lr_min", 1e-5)
        self.cfg.lr_max = getattr(self.cfg, "lr_max", 3e-3)
        self.cfg.epoch_min = getattr(self.cfg, "epoch_min", 4)
        self.cfg.epoch_max = getattr(self.cfg, "epoch_max", 32)
        self.cfg.kl_target = getattr(self.cfg, "kl_target", 0.01)

    # -----------------------------
    # Acting & memory
    # -----------------------------
    def act(self, state_np: np.ndarray, evaluation_mode: bool = False) -> int:
        """
        AC acting: 直接通过Actor输出
        """
        with torch.no_grad():
            s_np = np.asarray(state_np, dtype=np.float32)
            if s_np.ndim == 1:
                s_np = s_np[None, :]  # (1, obs_dim)
            s = torch.as_tensor(s_np, dtype=torch.float32, device=self.device)
            dist,value = self.net(s)  # [1, act_dim] # 似乎不用可以避免空间浪费，有空试试
            if evaluation_mode:
                # 测试 / eval：用“运用”策略，取概率最大的动作
                action = torch.argmax(dist.probs, dim=1)
            else:
                # 训练：按策略分布采样 → 自带探索
                action = dist.sample()  # shape [1]
                # self._last_state = s_np.copy() # 一系列state
                # self._last_action = int(action.item()) # Actor给出的一系列action
                self._last_logp = dist.log_prob(action)[0].item() # 计算PPO最关键的clip
                self._last_value = value[0].item() # critic给出的一系列value
            
            
            # calculate log_prob and V(s) of action based on current policy
        return int(action.item())
    

    def remember(self, state: np.ndarray, action: int, reward: float, done: bool, logp: float, value:float):
        """Store a single transition (s, a, r, s', done) into replay buffer."""
        # The ReplayBuffer's push() method handles squeezing [1, obs_dim] arrays
        self.memory.push(state, action, reward, done,logp, value)

    # -----------------------------
    # Learning from replay
    # -----------------------------
    def step(self, state: np.ndarray, action: int, reward: float, done: bool):
        """
        This is the main "learning" hook called by train.py
        1. Store the transition (s,a,r,s',done) in the replay buffer.
        2. Trigger one learning step (experience_replay) which samples from the buffer.
        """
        self.remember(state, action, reward, done, self._last_logp, self._last_value)
        # self.steps += 1
        
        if len(self.memory) > self.cfg.memory_size:
            self.experience_replay()
            self.memory.clear()


    def experience_replay(self):
        """
    
        """
        if len(self.memory) == 0:
            return
        
        # 1) 从buffer中取出所有数据
        
        # 2) Sample and convert to tensors
        s, a, r, m, logp_old, value = self.memory.get_all(self.cfg.memory_size)

        T = len(r) # 这一轮的步数
            # 转成 1D 张量
        s_t         = torch.as_tensor(s,         dtype=torch.float32, device=self.device)  # [T, obs_dim]
        a_t         = torch.as_tensor(a,         dtype=torch.int64,   device=self.device)  # [T]
        r_t         = torch.as_tensor(r,         dtype=torch.float32, device=self.device)  # [T]
        m_t         = torch.as_tensor(m,         dtype=torch.float32, device=self.device)  # [T]
        logp_old_t  = torch.as_tensor(logp_old,  dtype=torch.float32, device=self.device)  # [T]
        value_t     = torch.as_tensor(value,     dtype=torch.float32, device=self.device)  # [T]
        
        # 2) GAE计算advantage和returns      
        # 理论上 At = Q_st,at - V_st
        # 实际上Q可以选择使用前几步用真实r、后面步数用Critic评分组成，这里使用GAE加权构建  
        A_t = torch.zeros(T, dtype=torch.float32, device=self.device)
        returns  = torch.zeros(T, dtype=torch.float32, device=self.device)

        gae = 0.0
        
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0.0
            else:
                next_value = value_t[t + 1]

            delta = r_t[t] + m_t[t] * self.cfg.gamma * next_value - value_t[t]
            gae = delta + self.cfg.gamma * self.cfg.lambda_gae * gae
            A_t[t] = gae
            returns[t] = A_t[t] + value_t[t]
        
        # 对 advantage 标准化，有利于稳定训练
        A_t = (A_t - A_t.mean()) / (A_t.std() + 1e-8)
        
        if not torch.isfinite(A_t).all():
            print("Advantage has NaN/Inf")
            print(A_t)
            raise SystemExit
        
        for epoch in range(self.cfg.epoch):
            # 打乱T个下标
            indices = torch.randperm(T, device = self.device)
            for start in range(0, T, self.cfg.minibatch_size):
                end = start + self.cfg.minibatch_size
                mb_idx = indices[start:end] # 每次从数据集中抽出一段连续数据
                # 加载minibatch数据
                mb_s   = s_t[mb_idx]        # [B, obs_dim]
                mb_a   = a_t[mb_idx]        # [B]
                mb_logp_old = logp_old_t[mb_idx]  # [B]
                mb_adv = A_t[mb_idx]       # [B]
                mb_ret = returns[mb_idx]          # [B] 
                
                dist, value_pred = self.net(mb_s)
                logp = dist.log_prob(mb_a) # 现场更新，因为这是更新循环，第一次跑是一样的，后面更新就会和采样时得到的prob_old不一样
                entropy = dist.entropy().mean()
                
                if not torch.isfinite(logp).all():
                    print("logp has NaN/Inf: ", logp)
                    print("mb_a: ", mb_a)
                    raise SystemExit

                if not torch.isfinite(mb_logp_old).all():
                    print("old logp has NaN/Inf: ", mb_logp_old)
                    raise SystemExit

                ratio = torch.exp(logp - mb_logp_old)
                
                if not torch.isfinite(log_ratio).all():
                    print("log_ratio NaN/Inf:", log_ratio)
                    raise SystemExit

                # PPO-clip
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 
                                    1.0 - self.cfg.clip_eps, 
                                    1.0 + self.cfg.clip_eps) * mb_adv
                ##############actor loss!!!!
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # -------------Critic loss!!
                critic_loss = (value_pred - mb_ret).pow(2).mean()
                
                # -------------Total loss!!!
                loss = actor_loss + self.cfg.value_coef * critic_loss - self.cfg.entropy_coef * entropy
                
                # Backward
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
    
        

    # -----------------------------
    # Persistence
    # -----------------------------
    def save(self, path: str):
        """
        Save both online & target network weights plus config for reproducibility.
        Safe to version-control the small CartPole models.
        """
        torch.save(
            {
                "PPO": self.net.state_dict(),
                "cfg": self.cfg.__dict__,
            },
            path,
        )

    def load(self, path: str):
        """
        Load weights from disk onto the correct device.
        Note: Only loads weights; if you serialized optim state, add it here.
        """
        # For untrusted files, consider torch.load(..., weights_only=True) in future PyTorch
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["PPO"])
    
        # Optional: restore cfg from ckpt["cfg"] if you want to enforce same hyperparams
    def load_config(self, path: str | Path):
        """
        从 JSON 文件加载配置并应用到 agent。

        Args:
            path: JSON 配置文件的路径
        """
        path = Path(path) if isinstance(path, str) else path

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        try:
            with open(path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)

            # 创建新的配置对象，使用JSON中的值覆盖默认值
            new_cfg = PPOConfig()

            # 只更新JSON中存在的字段
            for key, value in config_dict.items():
                if hasattr(new_cfg, key):
                    # 确保类型匹配
                    current_type = type(getattr(new_cfg, key))
                    try:
                        # 尝试转换为正确的类型
                        if current_type == bool:
                            setattr(new_cfg, key, bool(value))
                        elif current_type == int:
                            setattr(new_cfg, key, int(value))
                        elif current_type == float:
                            setattr(new_cfg, key, float(value))
                        elif current_type == str:
                            setattr(new_cfg, key, str(value))
                        else:
                            setattr(new_cfg, key, value)
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Could not convert {key}={value} to {current_type}: {e}")
                        # 保持默认值
                else:
                    print(f"Warning: Unknown config key '{key}' in JSON file")

            # 更新agent的配置
            self.cfg = new_cfg

            # 更新设备设置
            if hasattr(self, 'device'):
                self.device = torch.device(self.cfg.device)
                # 将网络移到新设备
                self.online = self.online.to(self.device)
                self.target = self.target.to(self.device)

            print(f"[Info] Config loaded from {path}")
            print(f"[Info] New config: {self.cfg.__dict__}")

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {path}: {e}")
    # -----------------------------
    # Helpers
    # -----------------------------