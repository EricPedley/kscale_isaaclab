import torch
source = '/home/miller/IsaacLab/logs/skrl/quadcopter_direct/2025-10-09_22-03-24_ppo_torch/checkpoints/best_agent.pt'

model = torch.load(source)

# print summary
print(model)