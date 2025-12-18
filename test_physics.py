from agents.cartpole_physics import PhysicsAgent, PhysicsConfig
from train import evaluate_dqn

if __name__ == "__main__":
    print("=== 正在测试 Physics Agent (物理规则控制器) ===")
    cfg = PhysicsConfig(
        theta_coef=1.0,
        omega_coef=1.0,
        pos_coef=0.1,
        vel_coef=0.1
    )

    # 2. 实例化物理老师
    # CartPole-v1 的输入维度是 4，动作空间是 2
    teacher = PhysicsAgent(4, 2, cfg=cfg)

    # 3. 开始评估 (模仿你的 DQN 调用方式)
    # 注意：这里我们用 if_agent=True 直接传入 teacher 实例，
    # 这样就不需要去读取磁盘上的 JSON 文件了，非常方便。
    evaluate_dqn(
        algorithm="physics",  # 算法名，虽然这里仅仅用于打印日志
        episodes=100,         # 测试 100 局，看平均分是否稳定
        render=True,         # False=极速跑分; True=看画面
        fps=60,                # 0=全速运行
        if_agent=True,        # 关键：告诉函数我要直接传 agent 进去
        agent=teacher         # 把物理老师传进去
    )