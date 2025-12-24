from train import evaluate_agent
evaluate_agent(model_path=f"models/cartpole_dqn_1.00_0.00_128.00_61600.00_500.00_0.96_0.07_0.99_500.00.torch", algorithm="dqn", episodes=100, render=False, fps=60)




"""

python train.py --mode eval --path models/Best_weight_cartpole_dqn_12m6d18h26min.torch

"""