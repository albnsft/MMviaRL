import abc

from features.Features import baseState, State


class RewardFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculate(self, current_state: baseState, next_state: State) -> float:
        pass

    @abc.abstractmethod
    def reset(self):
        pass


class PnL(RewardFunction):
    def calculate(self, current_state: baseState, next_state: State) -> float:
        return next_state.portfolio.gain + next_state.portfolio.inventory *\
               (next_state.price - current_state.price)

    def reset(self):
        pass


class InventoryAdjustedPnL(RewardFunction):
    def __init__(self, inventory_aversion: float, asymmetrically_dampened: bool = False):
        self.inventory_aversion = inventory_aversion
        self.pnl = PnL()
        self.asymmetrically_dampened = asymmetrically_dampened

    def calculate(self, current_state: baseState, next_state: State) -> float:
        delta_midprice = next_state.price - current_state.price
        dampened_inventory_term = self.inventory_aversion * next_state.portfolio.inventory * delta_midprice
        if self.asymmetrically_dampened:
            dampened_inventory_term = max(0, dampened_inventory_term)
        return self.pnl.calculate(current_state, next_state) - dampened_inventory_term

    def reset(self):
        pass
