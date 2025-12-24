import numpy as np
import json
import os


class PhysicsConfig:

    def __init__(self,
                 theta_coef: float = 1.0,  # 角度权重
                 omega_coef: float = 1.0,  # 角速度权重
                 pos_coef: float = 0.0,  # 位置权重
                 vel_coef: float = 0.0,  # 速度权重
                 device: str = 'cpu'):  # 保持接口兼容性，实际不用
        self.theta_coef = theta_coef
        self.omega_coef = omega_coef
        self.pos_coef = pos_coef
        self.vel_coef = vel_coef
        self.device = device


class PhysicsAgent:
    """
    基于物理直觉的线性控制器。
    不使用神经网络，纯数学计算。
    """

    def __init__(self, obs_dim: int, act_dim: int, cfg: PhysicsConfig):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        # 将配置转化为权重向量 [pos, vel, theta, omega]
        self.weights = np.array([
            cfg.pos_coef,
            cfg.vel_coef,
            cfg.theta_coef,
            cfg.omega_coef
        ])

    def act(self, state):
        """
        根据物理公式决定动作。
        公式: Score = w1*x + w2*v + w3*theta + w4*omega
        """
        # 确保输入是扁平的 numpy 数组
        if hasattr(state, 'cpu'): state = state.cpu().numpy()  # 如果是Tensor
        state = np.array(state).flatten()

        # 计算线性加权和
        score = np.dot(state, self.weights)

        # 决策：大于0向右(1)，否则向左(0)
        return 1 if score > 0 else 0

    def step(self, state, action, reward, next_state, done):
        """物理 Agent 不需要训练步，此函数留空以保持接口一致"""
        pass

    def save(self, path):
        """保存参数到 JSON"""
        if path.endswith('.torch'):
            path = path.replace('.torch', '.json')

        data = {
            "theta_coef": self.cfg.theta_coef,
            "omega_coef": self.cfg.omega_coef,
            "pos_coef": self.cfg.pos_coef,
            "vel_coef": self.cfg.vel_coef
        }
        # 确保存储目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"[PhysicsAgent] Parameters saved to {path}")

    def load(self, path):
        """从 JSON 加载参数"""
        if path.endswith('.torch'):
            path = path.replace('.torch', '.json')

        with open(path, 'r') as f:
            data = json.load(f)

        self.cfg.theta_coef = data["theta_coef"]
        self.cfg.omega_coef = data["omega_coef"]
        self.cfg.pos_coef = data["pos_coef"]
        self.cfg.vel_coef = data["vel_coef"]

        # 更新权重
        self.weights = np.array([
            self.cfg.pos_coef,
            self.cfg.vel_coef,
            self.cfg.theta_coef,
            self.cfg.omega_coef
        ])
        print(f"[PhysicsAgent] Parameters loaded from {path}")