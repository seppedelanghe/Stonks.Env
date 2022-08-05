import math, os

from utils import terminal_screen

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from typing import Dict, Optional, Tuple
from objects import Stock, StockPrice

class StockModel(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super(StockModel, self).__init__()

        self.n_in = n_in
        self.n_out = n_out

        self.net = self._make_net()

    def randomize(self):
        for m in self.net:
            classname = m.__class__.__name__
            # for every Linear layer in a model
            if classname.find('Linear') != -1:
                # m.weight.data shoud be taken from a normal distribution
                m.weight.data.normal_(0.0,1/np.sqrt(self.n_in))
                # m.bias.data should be 0
                m.bias.data.fill_(0)

    def evolve(self, parent: nn.Module, polyak_factor: float):
        for target_param, param in zip(self.parameters(), parent.parameters()):
            target_param.data.copy_(polyak_factor * param.data + target_param.data * (1.0 - polyak_factor))

    def _make_net(self):
        return nn.Sequential(
            nn.Linear(self.n_in, self.n_in ** 4),
            nn.LeakyReLU(0.1),
            
            
            nn.Linear(self.n_in ** 4, self.n_in ** 2),
            nn.LeakyReLU(0.1),

            nn.Linear(self.n_in ** 2, self.n_out ** 2),
            nn.LeakyReLU(0.1),

            nn.Linear(self.n_out ** 2, self.n_out),
        )

    def forward(self, x):
        return self.net(x)


class StockMarket:
    def __init__(self):
        self.stocks: Dict[str, Stock] = {}
        self.time = 1 # days
        self.status = 0

    def add_stock(self, csv_dataset: str, ticker: str):
        self.stocks[ticker] = self._convert_csv(csv_dataset)

    def _convert_csv(self, path: str):
        data = pd.read_csv(path).dropna(axis=0)[['Open', 'Close', 'High', 'Low', 'Volume']].to_numpy(np.float32)
        return Stock(history=[StockPrice(open=row[0], close=row[1], low=row[2], high=row[3], volume=row[-1]) for row in data])

    @property
    def is_open(self):
        return self.status == 1

    def get_price(self, ticker: str):
        return self.stocks[ticker].get_price(self.time, self.status == 1)

    def get_day(self, ticker: str):
        return self.stocks[ticker].last if self.status == 2 else self.stocks[ticker].history[-2]

    def sell(self, ticker: str, amount: int = 1):
        price = self.get_price(ticker) * amount
        return price, amount

    def buy(self, ticker: str, amount: int = 1):
        price = self.get_price(ticker) * amount
        return price, amount

    def can_buy(self, ticker: str, funds: float, amount: int = 1):
        result = self.status == 1 and self.get_price(ticker) * amount <= funds
        return result

    def reset(self):
        self.time = 1
        self.status = 0

    def step(self):
        # if self.status == 0:
        #     self.status = 1 # open market
        # elif self.status == 1:
        #     self.status = 2 # close market
        # else:
        #     # reset + next day
        #     self.status = 0 
        self.status = 1
        self.time += 1


class StockEnv:
    def __init__(self, market: StockMarket, ticker: str, starting_funds: float = 1000.0):
        self.ticker = ticker
        self.market = market
        self.starting_funds = starting_funds
        self.funds = starting_funds
        self.owned = 10

    @property
    def max_iters(self):
        return len(self.market.stocks[self.ticker].history)

    @property
    def action_space(self):
        return (0, 1, 2)

    @property
    def n_actions(self):
        return len(self.action_space)

    @property
    def state(self):
        return (self.funds, self.owned, self.market.time, self.market.status, self.get_price())

    @property
    def n_observations(self):
        return len(self.state)

    @property
    def is_done(self):
        return self.market.time == len(self.market.stocks[self.ticker].history) - 1 or self.total_funds < self.starting_funds / 10

    @property
    def total_funds(self):
        return self.funds + self.market.get_price(self.ticker) * self.owned

    @property
    def reward(self):
        if self.owned < 0:
            raise Exception('f')
        return self.total_funds - self.starting_funds + (self.owned * 2)

    def reset(self):
        self.owned = 0
        self.funds = self.starting_funds

        return self.state

    def step(self, action: int = 0):
        if action == 1: # buy stock
            if self.market.can_buy(self.ticker, self.funds, action):
                cost, amount = self.market.buy(self.ticker)
                
                self.owned += amount
                self.funds -= cost

        elif action == 2 and self.owned > 0: # sell stock
            value, amount = self.market.sell(self.ticker)

            self.owned -= amount
            self.funds += value

        self.market.step()

    def get_price(self):
        return self.market.get_price(self.ticker)


class StockAgent:
    def __init__(self, market: StockMarket, ticker: str):
        self.market = market
        self.env = StockEnv(market, ticker)
        self.model = StockModel(self.env.n_observations, self.env.n_actions)

    def reset(self, model: bool = False):
        if model:
            self.model = StockModel(self.env.n_observations, self.env.n_actions)
        
        self.env.reset()
        self.market.reset()

    def run(self, limit: Optional[int] = None):
        state = self.env.reset()
        epochs = 0
        if type(limit) == type(None):
            limit = self.env.max_iters

        while not self.env.is_done and epochs < limit:        
            action = float(torch.argmax(self.model(torch.tensor(state)))) # take action that model suggests
            self.env.step(action)
            epochs += 1

        result = (epochs, self.env.reward)
        self.market.reset()
        return result
        

class StockAgentTrainer:
    def __init__(self, dataset_path: str, ticker: str, agents: int = 2):
        self.market = StockMarket()
        self.market.add_stock(dataset_path, ticker)
        self.ticker = ticker
        self.n_agents = agents

        self.agents: Tuple[StockAgent] = [StockAgent(self.market, ticker) for _ in range(agents)]

        self.random_factor = 0.99 # aka exploration
        self.lr = 0.001 # decreasement of random_factor
        self.limit_update_freq = 25


    def evolve(self, from_idx):
        randomize_idxs = np.rint(np.linspace(0, self.n_agents, math.ceil(self.n_agents * self.random_factor)))
        for idx, agent in enumerate(self.agents):
            if from_idx == idx:
                continue

            if idx in randomize_idxs:
                agent.model.randomize()
            else:
                agent.model.evolve(self.agents[from_idx].model, np.random.random())
            
            agent.reset() # reset market and funds

        # decay randomization
        self.random_factor -= (self.random_factor * self.lr)

    def randomize_all(self):
        for agent in self.agents:
            agent.model.randomize()
            agent.reset()

    def train(self, epochs: int = 10, limit_runs: int = 200):
        self.randomize_all()

        loop = tqdm(range(epochs), total=epochs, desc='epochs')
        for ep in loop:
            results = np.array([agent.run(limit_runs) for agent in tqdm(self.agents, total=self.n_agents, leave=False, desc='agents')])
            score = (results[:, 0] * 0.1) * results[:, 1]
            winner_idx = np.argmax(score)
            
            terminal_screen(results, ['episodes', 'funds'], 'agent', color_idx=winner_idx, ncols=10)

            loop.set_postfix(
                random_factor=self.random_factor,
                limit_runs=limit_runs
            )

            self.evolve(winner_idx)
            if ep % self.limit_update_freq == 0:
                limit_runs += math.ceil(limit_runs * 0.05)


if __name__ == "__main__":    
    trainer = StockAgentTrainer('/Users/seppe/Downloads/AAPL.csv', 'AAPL', 50)
    trainer.train(1000)