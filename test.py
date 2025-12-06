from train import evaluate_dqn
evaluate_dqn(model_path=f"models/Best_weight_cartpole_dqn_12m6d18h26min.torch", algorithm="dqn", episodes=100, render=False, fps=60)