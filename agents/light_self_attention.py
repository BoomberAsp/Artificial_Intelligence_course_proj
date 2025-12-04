import torch
import torch.nn as nn

class HybridQNet(nn.Module):
    """结合当前状态和历史特征的混合网络"""

    def __init__(self, obs_dim: int, act_dim: int, history_len: int = 5):
        super().__init__()

        # 处理当前状态（标准DQN路径）
        self.current_path = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # 处理历史特征（轻量注意力）
        self.history_path = nn.Sequential(
            nn.Linear(obs_dim * history_len, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # 融合和输出
        self.fusion = nn.Sequential(
            nn.Linear(16 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, act_dim)
        )

    def forward(self, current_state, history_states):
        # current_state: [B, obs_dim]
        # history_states: [B, history_len, obs_dim] 或展平

        # 当前状态路径
        current_features = self.current_path(current_state)

        # 历史路径（简单展平）
        batch_size = history_states.shape[0]
        history_flat = history_states.view(batch_size, -1)
        history_features = self.history_path(history_flat)

        # 融合
        combined = torch.cat([current_features, history_features], dim=1)
        q_values = self.fusion(combined)

        return q_values