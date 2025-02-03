import torch
import numpy as np
from prodRL.models.dqn_model import DQN
from prodRL.env_gym.product_allocation_env import ProductAllocationEnv
from prodRL.pipeline.train import train_model

inventory = [40, 72, 80, 41, 60, 87, 26, 53, 54, 56]
demand = [
    [25, 24, 39, 45, 34],
    [19, 34, 30, 28, 33],
    [21, 32, 2, 42, 14],
    [22, 48, 47, 38, 32],
    [29, 1, 9, 23, 36],
    [11, 14, 40, 15, 7],
    [41, 37, 3, 38, 18],
    [5, 34, 25, 40, 10],
    [9, 24, 46, 34, 8],
    [37, 20, 40, 18, 5],
]

model = train_model(inventory, demand)
torch.save(model.state_dict(), "models/saved_model.pth")


def load_model():
    model = DQN(10, 30)
    model.load_state_dict(torch.load("models/saved_model.pth"))
    model.eval()
    return model


model = load_model()


def allocate_products(inventory, demand):
    env = ProductAllocationEnv(inventory, demand)
    state = np.concatenate((inventory, demand))
    action = model(torch.tensor(state, dtype=torch.float)).detach().numpy()
    return action.tolist()


actions = allocate_products(inventory, demand)
print(actions)
