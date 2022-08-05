from typing import List
from pydantic import BaseModel
import random

class StockPrice(BaseModel):
    open: float
    close: float
    low: float
    high: float
    volume: int

class Stock(BaseModel):
    history: List[StockPrice]

    @property
    def last(self):
        return self.history[-1]

    def get_price(self, idx: int, market_open: bool = False):
        if not market_open:
            return self.history[idx].close

        # if no difference between high and low
        if int(self.history[idx].high * 100) - int(self.history[idx].low * 100) <= 1:
            return int(self.history[idx].high * 100)

        return random.randint(int(self.history[idx].low * 100), int(self.history[idx].high * 100)) / 100 # return random value between high and low
