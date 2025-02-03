# train.py
import torch
import numpy as np
from prodRL.models.dqn_model import DQN, get_optimizer
from prodRL.env_gym.product_allocation_env import ProductAllocationEnv
from prodRL.constant import HYPERPARAMETERS


def train_model(inventory_data, demand_data, episodes=1000):
    env = ProductAllocationEnv(inventory_data, demand_data)
    policy_net = DQN(env.observation_space.shape[0], np.prod(env.action_space.nvec)).to(
        "cpu"
    )
    optimizer = get_optimizer(policy_net, HYPERPARAMETERS["learning_rate"])
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = np.random.randint(0, np.min([env.inventory, env.demand], axis=0))
            next_state, reward, done, _ = env.step(action)
            optimizer.zero_grad()
            loss = torch.tensor(reward, dtype=torch.float)
            loss.backward()
            optimizer.step()
    return policy_net
