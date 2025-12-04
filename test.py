from train import evaluate_dqn
evaluate_dqn(model_path=f"models/cartpole_dqn_12m4d15h23min.torch", algorithm="dqn", episodes=100, render=False, fps=60)