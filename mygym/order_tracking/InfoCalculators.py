import abc

import numpy as np
import pandas as pd
from numpy import ndarray
from copy import deepcopy

from features.Features import State
from rewards.RewardFunctions import RewardFunction


class _InfoCalculator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculate(self, internal_state: State, action: np.ndarray):
        pass


class InfoCalculator(_InfoCalculator):
    def __init__(
            self,
            verbose: bool = True
    ):
        self.verbose = verbose

    def reset_episode(self):
        self.spreads = []
        self.inventories = []
        self.pnls, self.pnl = [], 0
        self.actions = {'tetha buy': [], 'tetha sell': []}
        self.filled_actions = deepcopy(self.actions)
        self.aums, self.aum = [], 0
        self.nd_pnl = 0
        self.map = 0
        self.dates = []
        self.mid_price = []

    def calculate(self, internal_state: State, reward_relative_midprice: RewardFunction) -> pd.DataFrame:
        self._update_args(reward_relative_midprice)
        self._update_lists(internal_state)
        self._update_metrics(internal_state)
        filled, info = self._compute_filled(internal_state), self._compute_info(internal_state)
        if self.verbose and np.any(filled!=0):
            print('*' * 50)
            print(filled)
            print('\n')
            print(info.T)
            print('*' * 50)
        return info

    def _update_args(self, reward_relative_midprice: RewardFunction):
        self.pnl += reward_relative_midprice

    def _update_lists(self, internal_state: State):
        self.mid_price.append(internal_state.orderbook.midprice)
        self.spreads.append(internal_state.orderbook.spread)
        self.inventories.append(internal_state.portfolio.inventory)
        self.pnls.append(self.pnl)
        self.dates.append(internal_state.now_is)
        self.actions['tetha buy'].append(internal_state.buy_parameter)
        self.actions['tetha sell'].append(internal_state.sell_parameter)

    def _update_metrics(self, internal_state: State):
        self.nd_pnl = self.calculate_nd_pnl()
        self.map = self.calculate_map()
        self.aum = self.calculate_aum(internal_state)
        self.aums.append(self.aum)

    def _compute_filled(self, internal_state: State) -> pd.DataFrame:
        col = [["buy", "sell"], ["price", "volume"]]
        date = internal_state.filled_orders.internal[0].timestamp if len(
            internal_state.filled_orders.internal) > 0 else None
        index = pd.MultiIndex.from_product(col, names=[date, ""])
        filled = pd.DataFrame(np.zeros((4, 1)), index=index, columns=["agent's filled orders"])
        for order in internal_state.filled_orders.internal:
            filled.loc[order.direction] = np.array([order.price, order.volume]).reshape(-1, 1)
        if internal_state.buy_parameter == 0 and internal_state.sell_parameter == 0:
            self.filled_actions['tetha buy'].append(internal_state.buy_parameter)
            self.filled_actions['tetha sell'].append(internal_state.sell_parameter)
        elif np.any(filled.loc["buy"]!=0) and np.any(filled.loc["sell"]!=0):
            self.filled_actions['tetha buy'].append(internal_state.buy_parameter)
            self.filled_actions['tetha sell'].append(internal_state.sell_parameter)
        elif np.any(filled.loc["buy"]!=0) and np.any(filled.loc["sell"]==0):
            self.filled_actions['tetha buy'].append(internal_state.buy_parameter)
            self.filled_actions['tetha sell'].append(-1)
        elif np.any(filled.loc["buy"]==0) and np.any(filled.loc["sell"]!=0):
            self.filled_actions['tetha buy'].append(-1)
            self.filled_actions['tetha sell'].append(internal_state.sell_parameter)
        return filled

    def _compute_info(self, internal_state: State) -> pd.DataFrame:
        pnl_mp_pe = (self.pnls[-1]-self.pnls[-2]) if len(self.pnls)>1 else self.pnls[0]
        info_dict = dict(
            spread=internal_state.orderbook.spread,
            mid_price=internal_state.price,
            tetha_sell=internal_state.sell_parameter,
            tetha_buy=internal_state.buy_parameter,
            pnl_per_episode=pnl_mp_pe,
            normalised_pnl=self.nd_pnl,
            inventory=self.inventories[-1],
            inventory_ma=self.map,
            aum=self.aum,
        )
        info = pd.DataFrame([info_dict], index=[internal_state.now_is])
        return info

    def calculate_aum(self, internal_state: State) -> float:
        return internal_state.portfolio.cash + internal_state.price * internal_state.portfolio.inventory

    def calculate_nd_pnl(self) -> float:
        return self.pnl / np.mean(self.spreads)

    def calculate_map(self) -> ndarray:
        return np.mean(np.abs(self.inventories))