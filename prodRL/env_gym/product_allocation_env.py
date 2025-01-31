# product_allocation_env.py
import gym
import numpy as np
from gym import spaces


class ProductAllocationEnv(gym.Env):
    def __init__(self, inventory_data, demand_data):
        super(ProductAllocationEnv, self).__init__()
        self.inventory = inventory_data
        self.demand = demand_data
        self.num_products = len(inventory_data)
        self.num_shops = len(demand_data)
        self.observation_space = spaces.Box(
            low=0,
            high=np.max(inventory_data),
            shape=(self.num_products + self.num_shops,),
            dtype=np.int32,
        )
        self.action_space = spaces.MultiDiscrete(
            [np.max(inventory_data)] * (self.num_products * self.num_shops)
        )
        self.state = np.concatenate((self.inventory, self.demand))

    def step(self, action):
        allocation = np.array(action).reshape(self.num_products, self.num_shops)
        reward = 0
        for p in range(self.num_products):
            for s in range(self.num_shops):
                allocated = min(allocation[p, s], self.inventory[p], self.demand[s])
                self.inventory[p] -= allocated
                self.demand[s] -= allocated
                reward += allocated
        self.state = np.concatenate((self.inventory, self.demand))
        done = np.all(self.demand <= 0) or np.all(self.inventory <= 0)
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.concatenate((self.inventory, self.demand))
        return self.state
