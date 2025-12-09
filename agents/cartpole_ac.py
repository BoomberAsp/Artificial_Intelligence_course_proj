"""
PyTorch AC for CartPole (Gymnasium)
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


# -----------------------------
# Default Hyperparameters
# -----------------------------
GAMMA = 0.9                    # reward discount in TD error
VALUE_COEF = 0.5        # critic loss 权重
ENTROPY_COEF = 1e-3     # 熵正则，鼓励探索
LR = 0.001                    # learning rate for critic
BATCH_SIZE = 32
MEMORY_SIZE = 5000


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
    - Each entry stores (s, a, r, s', mask) where mask = 0 if terminal else 1.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buf: Deque[Tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        s = np.asarray(s)
        s2 = np.asarray(s2)
        # Squeeze arrays that are [1, obs_dim] down to [obs_dim] for storage
        if s.ndim == 2 and s.shape[0] == 1:
            s = s.squeeze(0)
        if s2.ndim == 2 and s2.shape[0] == 1:
            s2 = s2.squeeze(0)
        self.buf.append((s, a, r, s2, 0.0 if done else 1.0))

    def sample(self, batch_size: int):
        # Uniformly sample a mini-batch of transitions
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, m = zip(*batch)
        # Shapes after stacking:
        #  s, s2: [B, obs_dim], a: [B], r: [B], m: [B]
        return (
            np.stack(s, axis=0),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(s2, axis=0),
            np.array(m, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


@dataclass
class ACConfig:
    """
    # class PPOConfig:
    #   - gamma, lr, rollout_steps, update_epochs, minibatch_size
    #   - clip_eps, value_coef, entropy_coef, lambda_gae
    #   - device 同样保留

    """
    lr: float = LR
    batch_size: int = BATCH_SIZE
    memory_size: int = MEMORY_SIZE
    gamma : float = GAMMA                    # reward discount in TD error
    value_coef : float = VALUE_COEF        # critic loss 权重
    entropy_coef : float = ENTROPY_COEF    # 熵正则，鼓励探索

    # Auto-select CUDA if available; CPU is perfectly fine for CartPole
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ACSolver:
    """
    the following explanation:
    - self.net: ActorCritic 网络，输出 (logits, V(s))
    - self.optim: Adam 优化器，优化 actor 和 critic 所有参数
    - self.memory: ReplayBuffer，用来存当前这一轮 rollout 的数据
    - self.steps: 总交互步数计数
    """

    def __init__(self, observation_space: int, action_space: int, cfg: ACConfig | None = None):
        # Store dimensions and hyperparameters
        self.obs_dim = observation_space
        self.act_dim = action_space
        self.cfg = cfg or ACConfig()

        # Choose device (GPU if available, else CPU)
        self.device = torch.device(self.cfg.device)

        # Generate ActorCritic network
        self.net = ActorCritic(self.obs_dim, self.act_dim).to(self.device)

        # Optimizer over online network parameters
        self.optim = optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        
        # Experience replay memory        
        self.memory = ReplayBuffer(self.cfg.memory_size)

        # Global counters
        self.steps = 0

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
        return int(action.item())
    

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Store a single transition (s, a, r, s', done) into replay buffer."""
        # The ReplayBuffer's push() method handles squeezing [1, obs_dim] arrays
        self.memory.push(state, action, reward, next_state, done)

    # -----------------------------
    # Learning from replay
    # -----------------------------
    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        This is the main "learning" hook called by train.py
        1. Store the transition (s,a,r,s',done) in the replay buffer.
        2. Trigger one learning step (experience_replay) which samples from the buffer.
        """
        self.remember(state, action, reward,  next_state, done)
        # self.experience_replay()

    def experience_replay(self):
        """
        
        """
        # # 1) Warmup and capacity check: skip updates if insufficient data
        # if len(self.memory) < max(self.cfg.batch_size, self.cfg.initial_exploration):
        #     return

        # 2) Sample and convert to tensors
        s, a, r, s2, m = self.memory.sample(self.cfg.batch_size)

        s_t  = torch.as_tensor(s,  dtype=torch.float32, device=self.device)               # [B, obs_dim]
        a_t  = torch.as_tensor(a,  dtype=torch.int64,   device=self.device).unsqueeze(1) # [B, 1]
        r_t  = torch.as_tensor(r,  dtype=torch.float32, device=self.device).unsqueeze(1) # [B, 1]
        s2_t = torch.as_tensor(s2, dtype=torch.float32, device=self.device)               # [B, obs_dim]
        m_t  = torch.as_tensor(m,  dtype=torch.float32, device=self.device).unsqueeze(1) # [B, 1]; 0 if done else 1

        # 加载数据
        dist, value = self.net(s_t)
        action = a_t
        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        
        value = value.squeeze(0)

        # Compute target values using the target network (no gradient)
        with torch.no_grad():
            _, next_value = self.net(s2_t)      # [B]
            next_value = next_value.unsqueeze(1)  # [B,1] if你前面都用 [B,1]
        # 计算TD error
        delta = r_t + self.cfg.gamma * m_t * next_value - value  # 都转为 [B,1] 或 [B]
                
        
        # -------------Critic loss!!
        critic_loss = delta.pow(2).mean()
        
        # -------------Actor loss!!
        actor_loss = -(log_prob * delta.detach()).mean()
        
        # -------------熵正则
        loss = actor_loss + self.cfg.value_coef * critic_loss - self.cfg.entropy_coef * entropy

        # 5) Backpropagation step
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
        # 6) Record
        self.steps += 1

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
                "ActorCritic": self.net.state_dict(),
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
        self.net.load_state_dict(ckpt["ActorCritic"])
    
        # Optional: restore cfg from ckpt["cfg"] if you want to enforce same hyperparams

    # -----------------------------
    # Helpers
    # -----------------------------