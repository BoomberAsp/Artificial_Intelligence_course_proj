from agents.cartpole_physics import PhysicsAgent, PhysicsConfig
from train import evaluate_agent

if __name__ == "__main__":
    print("=== 正在测试 Physics Agent (物理规则控制器) ===")

    # 1. 配置参数
    cfg = PhysicsConfig(
        theta_coef=1.0,
        omega_coef=1.0,
        pos_coef=0.1,
        vel_coef=0.1
    )

    # 2. 实例化物理老师
    # CartPole-v1 的输入维度是 4，动作空间是 2
    teacher = PhysicsAgent(4, 2, cfg=cfg)

    # 3. 开始评估
    evaluate_agent(
        algorithm="physics",  # 算法名
        episodes=100,  # 测试 100 局
        render=False,  # True=看画面
        fps=0,  # 限制帧率以便观察
        if_agent=True,  # [关键] 告诉函数直接使用传入的 agent 实例
        agent=teacher  # 传入 teacher 实例
    )