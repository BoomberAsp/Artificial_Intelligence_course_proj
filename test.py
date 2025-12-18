from train import evaluate_agent
evaluate_agent(model_path=f"models/Best_weight_cartpole_dqn_12m6d18h26min.torch", algorithm="dqn", episodes=100, render=False, fps=60)




"""

python train.py --mode eval --path models/Best_weight_cartpole_dqn_12m6d18h26min.torch

"""