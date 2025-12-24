from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import agents.cartpole_dqn
from agents.cartpole_dqn import ReplayBuffer,DQNConfig,DQNSolver


PAPER_CONFIG = {
    "lr": 0.00025,
    "gamma": 0.99,
    "batch_size": 32,
    "memory_size": 50000,
    "target_update": 1000,
    "eps_start": 1.0,
    "eps_end": 0.01,
    "eps_decay": 0.999,
    "initial_exploration": 1000,
    "alpha": 0.1,
    "beta": 0.4,
    "beta_increment": 0.001
}


class PriorityReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        super().__init__(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        # 使用list而不是deque来存储优先级，便于索引操作
        self.priorities = []
        self._max_priority = 1.0

    def push(self, s, a, r, s2, done):
        super().push(s, a, r, s2, done)
        # 新经验以最高优先级存入
        if len(self.priorities) < self.capacity:
            self.priorities.append(self._max_priority)
        else:
            # 当缓冲区满时，移除最旧的优先级
            self.priorities.pop(0)
            self.priorities.append(self._max_priority)

    def sample(self, batch_size: int):
        if len(self.buf) == 0:
            raise ValueError("The replay buffer is empty.")

        if len(self.buf) < batch_size:
            batch_size = len(self.buf)  # 确保不会采样超过缓冲区大小

        # 计算采样概率
        priorities = np.array(self.priorities[:len(self.buf)], dtype=np.float64)  # 使用float64避免数值问题
        probs = priorities ** self.alpha
        probs = probs / (probs.sum() + 1e-8)  # 加小值防止除零

        # 根据概率采样索引
        indices = np.random.choice(len(self.buf), batch_size, p=probs, replace=False)

        # 计算重要性采样权重
        total = len(self.buf)
        weights = (total * probs[indices]) ** (-self.beta)
        weights = weights / weights.max() if len(weights) > 0 else weights
        weights = np.array(weights, dtype=np.float32)

        # 获取采样数据
        batch = [self.buf[idx] for idx in indices]
        s, a, r, s2, m = zip(*batch)

        return (
            np.stack(s, axis=0),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(s2, axis=0),
            np.array(m, dtype=np.float32),
            indices,
            weights,
        )

    def update_priorities(self, indices, td_errors):
        """更新采样经验的优先级"""
        td_errors = np.abs(td_errors) + 1e-6  # 避免零优先级，使用更小的epsilon

        for idx, error in zip(indices, td_errors):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = float(error)

        # 更新最大优先级
        if len(self.priorities) > 0:
            self._max_priority = max(self.priorities)

        # 增加beta（逐渐降低重要性采样纠正程度）
        self.beta = min(1.0, self.beta + self.beta_increment)


class PDQNConfig(DQNConfig):
    def __init__(self, **kwargs):
        # 调用父类初始化
        super().__init__(**kwargs)
        self.alpha = kwargs.pop('alpha', 0.6)
        self.beta = kwargs.pop('beta', 0.4)  # 从0.4开始
        self.beta_increment = kwargs.pop('beta_increment', 0.001)


        self.alpha = PAPER_CONFIG.get('alpha', 0.6)
        self.beta = PAPER_CONFIG.get('beta', 0.4)
        self.beta_increment = PAPER_CONFIG.get('beta_increment', 0.001)
        self.batch_size = PAPER_CONFIG.get('batch_size', 32)
        self.memory_size = PAPER_CONFIG.get('memory_size', 50000)
        self.target_update = PAPER_CONFIG.get('target_update', 1000)
        self.eps_start = PAPER_CONFIG.get('eps_start', 1.0)
        self.eps_end = PAPER_CONFIG.get('eps_end', 0.01)
        self.eps_decay = PAPER_CONFIG.get('eps_decay', 0.999)
        self.initial_exploration = PAPER_CONFIG.get('initial_exploration', 1000)
        self.alpha = PAPER_CONFIG.get('alpha', self.alpha)
        self.beta = PAPER_CONFIG.get('beta', self.beta)
        self.beta_increment = PAPER_CONFIG.get('beta_increment', 0.001)
        self.lr = PAPER_CONFIG.get('lr', 0.001)

        self.replay_buffer_type = PriorityReplayBuffer
        self.replay_buffer_params = {
            'alpha': self.alpha,
            'beta': self.beta,
            'beta_increment': self.beta_increment
        }





class PDQNSolver(DQNSolver):
    """带优先级经验回放的DQN求解器"""

    def __init__(self, observation_space: int, action_space: int, cfg: PDQNConfig | None = None):
        # 使用PDQNConfig作为默认配置
        self.cfg = cfg or PDQNConfig()

        # 调用父类初始化，但覆盖记忆缓冲区
        super().__init__(observation_space, action_space, self.cfg)

        # 创建优先级经验回放缓冲区
        self.memory = self.cfg.replay_buffer_type(
            self.cfg.memory_size,
            **self.cfg.replay_buffer_params
        )

    def experience_replay(self):
        """重写经验回放方法以支持优先级采样"""
        # 1) Warmup和容量检查
        if len(self.memory) < max(self.cfg.batch_size, self.cfg.initial_exploration):
            self._decay_eps()
            return

        # 2) 从优先级缓冲区采样
        s, a, r, s2, m, indices, weights = self.memory.sample(self.cfg.batch_size)

        # 转换为张量
        s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a_t = torch.as_tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r_t = torch.as_tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        s2_t = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        m_t = torch.as_tensor(m, dtype=torch.float32, device=self.device).unsqueeze(1)
        weights_t = torch.as_tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        # 3) 计算当前Q值
        q_sa = self.online(s_t).gather(1, a_t)

        # 4) 计算目标Q值
        with torch.no_grad():
            q_next = self.target(s2_t).max(dim=1, keepdim=True)[0]
            target = r_t + m_t * self.cfg.gamma * q_next

        # 5) 计算TD误差和损失（带重要性采样权重）
        td_errors = target - q_sa

        # Huber损失比MSE更稳定
        criterion = nn.SmoothL1Loss(reduction='none')
        elementwise_loss = criterion(q_sa, target)
        loss = (weights_t * elementwise_loss).mean()

        # 6) 反向传播
        self.optim.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_value_(self.online.parameters(), 1.0)

        self.optim.step()

        # 7) 更新采样经验的优先级
        with torch.no_grad():
            # 使用当前TD误差的绝对值更新优先级
            td_errors_np = td_errors.abs().squeeze().cpu().numpy()
        self.memory.update_priorities(indices, td_errors_np)

        # 8) 探索率衰减
        self._decay_eps()

        # 9) 更新目标网络
        if self.steps % self.cfg.target_update == 0:
            self.update_target(hard=True)

    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """重写step方法，使用优先级缓冲区"""
        # 存储经验
        self.remember(state, action, reward, next_state, done)

        # 更新计数器
        self.steps += 1

        # 如果缓冲区足够大，进行学习
        if len(self.memory) >= self.cfg.initial_exploration:
            self.experience_replay()


# 添加一个用于兼容的函数，可以在train.py中使用
def create_pdqn_solver(observation_space: int, action_space: int, cfg: PDQNConfig | None = None):
    """创建PDQN求解器的工厂函数"""
    return PDQNSolver(observation_space, action_space, cfg)


if __name__ == "__main__":
    # 简单测试
    config = PDQNConfig()
    config.memory_size = 1000
    config.batch_size = 32

    solver = PDQNSolver(4, 2, config)
    print("PDQN Solver created successfully")
    print(f"Memory type: {type(solver.memory)}")