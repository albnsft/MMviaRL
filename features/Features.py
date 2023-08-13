from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, time, timedelta, date

import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import abc

from orderbook.models import Orderbook, FilledOrders

from typing import Optional


class CannotUpdateError(Exception):
    pass


@dataclass
class Portfolio:
    inventory: int
    cash: float
    gain: float


@dataclass
class baseState:
    price: float #midprice_orderbook
    portfolio: Portfolio

@dataclass
class State:
    filled_orders: FilledOrders
    orderbook: Orderbook
    price: float #midprice_orderbook
    portfolio: Portfolio
    now_is: datetime
    buy_parameter: int
    sell_parameter: int


class Feature(metaclass=abc.ABCMeta):
    def __init__(
        self,
        name: str,
        min_value: float,
        max_value: float,
        update_frequency: timedelta,
        lookback_periods: int,
        normalisation_on: bool,
        max_norm_len: int = 100000,
    ):
        self.name = name
        self.min_value = min_value
        self.max_value = max_value
        assert update_frequency <= timedelta(minutes=1), "HFT update frequency must be less than 1 minute."
        self.update_frequency = update_frequency
        self.lookback_periods = lookback_periods
        self.normalisation_on = normalisation_on
        self.max_norm_len = max_norm_len
        self.current_value = 0.0
        self.first_usage_time = datetime.min
        self.scalar = MinMaxScaler([-1, 1])
        if self.normalisation_on:
            self.history: deque = deque(maxlen=max_norm_len)

    @property
    def window_size(self) -> timedelta:
        return self.lookback_periods * self.update_frequency

    def normalise(self, value: float) -> float:
        if len(self.history) == 0:
            # To prevent a Nan value from being returned
            # if the queue is empty:
            self.history.append(value + 1e-06)
        self.history.append(value)
        return self.scalar.fit_transform(np.array(self.history).reshape(-1, 1)).squeeze()[-1] #StandardScaler().fit_transform(np.array(self.history).reshape(-1, 1)).squeeze()[-1] stats.zscore(self.history)[-1]

    @abc.abstractmethod
    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        pass

    def update(self, state: State) -> None:
        if state.now_is >= self.first_usage_time and self._now_is_multiple_of_update_freq(state.now_is):
            self._update(state)
            value = self.clamp(self.current_value, min_value=self.min_value, max_value=self.max_value)
            if value != self.current_value:
                print(f"Clamping value of {self.name} from {self.current_value} to {value}.")
            self.current_value = self.normalise(value) if self.normalisation_on else value

    @abc.abstractmethod
    def _update(self, state: State) -> None:
        pass

    def _reset(self, state: State, first_usage_time: Optional[datetime] = None):
        self.first_usage_time = first_usage_time or datetime.min
        if self.normalisation_on:
            self.history.clear()
        self._update(state)

    @staticmethod
    def clamp(number: float, min_value: float, max_value: float):
        return max(min(number, max_value), min_value)

    def _now_is_multiple_of_update_freq(self, now_is: datetime):
        return timedelta(seconds=now_is.second, microseconds=now_is.microsecond) % self.update_frequency == timedelta(
            microseconds=0
        )


########################################################################################################################
#                                                   Book features                                                      #
########################################################################################################################


class Spread(Feature):
    def __init__(
        self,
        name: str = "Spread",
        min_value: float = 0,
        max_value: float = (50 * 100),  # 50 ticks
        update_frequency: timedelta = timedelta(seconds=1),
        normalisation_on: bool = False
    ):
        super().__init__(name, min_value, max_value, update_frequency, 0, normalisation_on)

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        super()._reset(state, first_usage_time)

    def _update(self, state: State) -> None:
        self.current_value = state.orderbook.spread


class DeltaMidPrice(Feature):
    def __init__(
        self,
        update_frequency: timedelta = timedelta(seconds=1),
        normalisation_on: bool = False
    ):
        super().__init__("DeltaMidPrice", -1, 1, update_frequency, 0, normalisation_on)

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        super()._reset(state, first_usage_time)

    def _update(self, state: State) -> None:
        self.current_value = state.orderbook.midprice


class BookImbalance(Feature):
    def __init__(
        self,
        update_frequency: timedelta = timedelta(seconds=1),
        normalisation_on: bool = False,
    ):
        super().__init__("BookImbalance", -1, 1, update_frequency, 0, normalisation_on)

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        super()._reset(state, first_usage_time)

    def _update(self, state: State) -> None:
        self.current_value = state.orderbook.imbalance


########################################################################################################################
#                                                  Price features                                                      #
########################################################################################################################


class PriceMove(Feature):
    """The price move is calculated as the difference between the price at time now_is - min_updates * update_freq
    and now_is. Here, the price is given when calling update and could be defined as the midprice, the microprice, or
    any other sensible notion of price."""

    def __init__(
        self,
        name: str = "MidpriceMove",
        min_value: float = -100 * 100,  # 100 tick downward move
        max_value: float = 100 * 100,  # 100 tick upward move
        update_frequency: timedelta = timedelta(seconds=1),
        lookback_periods: int = 1,  # Calculate the move in the midprice between 10 time periods ago and now
        normalisation_on: bool = False,
    ):
        super().__init__(name, min_value, max_value, update_frequency, lookback_periods, normalisation_on)
        self.prices: deque = deque(maxlen=self.lookback_periods + 1)

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        self.prices = deque(maxlen=self.lookback_periods + 1)
        super()._reset(state, first_usage_time)

    def _update(self, state: State) -> None:
        self.prices.appendleft(state.price)
        self.current_value = self.prices[0] - self.prices[-1]


class Volatility(Feature):
    """The volatility of the midprice series over a trailing window. We use the variance of percentage returns as
    opposed to the standard deviation of percentage returns as variance scales linearly with time and is therefore more
    reasonably a dimensionless attribute of the returns series. Furthermore, we ignore the mean of the returns since
    they are too noisy an observation and a *much* larger number of returns is required for it to be useful."""

    def __init__(
        self,
        name: str = "Volatility",
        min_value: float = 0,
        max_value: float = 1.0,
        update_frequency: timedelta = timedelta(seconds=1),
        lookback_periods: int = 10,
        normalisation_on: bool = False,
    ):
        super().__init__(name, min_value, max_value, update_frequency, lookback_periods, normalisation_on)
        self.prices: deque = deque(maxlen=self.lookback_periods + 1)

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        self.prices = deque(maxlen=self.lookback_periods + 1)
        super()._reset(state, first_usage_time)

    def _update(self, state: State) -> None:
        if len(self.prices) < self.lookback_periods:
            self.prices.append(state.price)
            self.current_value = 0.0
        elif len(self.prices) >= self.lookback_periods:
            self.prices.append(state.price)
            pct_returns = np.diff(np.array(self.prices)) / np.array(self.prices)[1:]
            self.current_value = sum(pct_returns**2) / self.lookback_periods


class RSI(Feature):
    """The relative strength index (RSI) is a momentum indicator used in technical analysis, measures the speed and
     magnitude of a security's recent price changes to evaluate overvalued or undervalued conditions
    in the price of that security"""

    def __init__(
        self,
        name: str = "RSI",
        min_value: float = 0,
        max_value: float = 100,
        update_frequency: timedelta = timedelta(seconds=1),
        lookback_periods: int = 10,
        normalisation_on: bool = False
    ):
        super().__init__(name, min_value, max_value, update_frequency, lookback_periods, normalisation_on)
        self.prices: deque = deque(maxlen=self.lookback_periods + 1)

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        self.prices = deque(maxlen=self.lookback_periods + 1)
        super()._reset(state, first_usage_time)

    def _update(self, state: State) -> None:
        if len(self.prices) < self.lookback_periods:
            self.prices.append(state.price)
            self.current_value = 0.0
        elif len(self.prices) >= self.lookback_periods:
            self.prices.append(state.price)
            pct_returns = np.diff(np.array(self.prices)) / np.array(self.prices)[1:]
            if np.any(np.where(pct_returns > 0)[0]) and np.any(np.where(pct_returns < 0)[0]):
                avg_up = np.mean(pct_returns[np.where(pct_returns > 0)])
                avg_down = np.abs(np.mean(pct_returns[np.where(pct_returns < 0)]))
                self.current_value = 100 * avg_up/(avg_up+avg_down)
            else:
                self.current_value = 0

########################################################################################################################
#                                                Order Flow features                                                   #
########################################################################################################################


class TradeDirectionImbalance(Feature):
    """The trade direction imbalance is given by number_buy_executions - number_sell_executions / total_executions
    over a given period."""

    def __init__(
        self,
        name: str = "TradeImbalance",
        update_frequency: timedelta = timedelta(seconds=1),
        lookback_periods: int = 600,
        track_internal: bool = False,
        normalisation_on: bool = False
    ):
        super().__init__(name, -1.0, 1.0, update_frequency, lookback_periods, normalisation_on)
        self.track_internal = track_internal
        self.trades: dict = dict(buy=deque(maxlen=self.lookback_periods), sell=deque(maxlen=self.lookback_periods))
        self.total_trades = 0
        self.trade_diff = 0

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        self.trades = dict(buy=deque(maxlen=self.lookback_periods), sell=deque(maxlen=self.lookback_periods))
        self.total_trades = 0
        self.trade_diff = 0
        super()._reset(state, first_usage_time)

    def _update(self, state: State) -> None:
        num_buys = sum(1 for order in state.filled_orders.external if order.direction == "buy")
        num_sells = sum(1 for order in state.filled_orders.external if order.direction == "sell")
        if self.track_internal:
            num_buys += sum(1 for order in state.filled_orders.internal if order.direction == "buy")
            num_sells += sum(1 for order in state.filled_orders.internal if order.direction == "sell")
        if len(self.trades["buy"]) < self.lookback_periods:
            self._update_trades(num_buys, num_sells)
            self.current_value = 0.0
        elif self.total_trades == 0:
            self._update_trades(num_buys, num_sells)
            self.total_trades = sum(self.trades["buy"]) + sum(self.trades["sell"])
            self.trade_diff = sum(self.trades["buy"]) - sum(self.trades["sell"])
            self._update_current_value()
        else:
            oldest_trades = {side: self.trades[side].popleft() for side in ("buy", "sell")}
            self.total_trades -= oldest_trades["buy"] + oldest_trades["sell"]
            self.total_trades += num_buys + num_sells
            self.trade_diff -= oldest_trades["buy"] - oldest_trades["sell"]
            self.trade_diff += num_buys - num_sells
            self._update_trades(num_buys, num_sells)
            self._update_current_value()

    def _update_current_value(self):
        if self.total_trades != 0:
            self.current_value = self.trade_diff / self.total_trades
        else:
            self.current_value = 1 / 2

    def _update_trades(self, num_buys: int, num_sells: int):
        self.trades["buy"].append(num_buys)
        self.trades["sell"].append(num_sells)


class TradeVolumeImbalance(Feature):
    """The trade volume imbalance is given by volume_buy_executions - volume_sell_executions / total_volume_executions
    over a given period."""

    def __init__(
        self,
        name: str = "TradeVolumeImbalance",
        update_frequency: timedelta = timedelta(seconds=1),
        lookback_periods: int = 600,
        track_internal: bool = False,
        normalisation_on: bool = False
    ):
        super().__init__(name, -1.0, 1.0, update_frequency, lookback_periods, normalisation_on)
        self.track_internal = track_internal
        self.volumes: dict = dict(buy=deque(maxlen=self.lookback_periods), sell=deque(maxlen=self.lookback_periods))
        self.total_volume = 0
        self.volume_imbalance = 0

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        self.volumes = dict(buy=deque(maxlen=self.lookback_periods), sell=deque(maxlen=self.lookback_periods))
        self.total_volume = 0
        self.volume_imbalance = 0
        super()._reset(state, first_usage_time)

    def _update(self, state: State) -> None:
        buy_volume = sum(order.volume for order in state.filled_orders.external if order.direction == "buy")
        sell_volume = sum(order.volume for order in state.filled_orders.external if order.direction == "sell")
        if self.track_internal:
            buy_volume += sum(order.volume for order in state.filled_orders.internal if order.direction == "buy")
            sell_volume += sum(order.volume for order in state.filled_orders.internal if order.direction == "sell")
        if len(self.volumes["buy"]) < self.lookback_periods:
            self._update_volumes(buy_volume, sell_volume)
            self.current_value = 0.0
        elif self.total_volume == 0:
            self._update_volumes(buy_volume, sell_volume)
            self.total_volume = sum(self.volumes["buy"]) + sum(self.volumes["sell"])
            self.volume_imbalance = sum(self.volumes["buy"]) - sum(self.volumes["sell"])
            self._update_current_value()
        else:
            oldest_volumes = {side: self.volumes[side].popleft() for side in ("buy", "sell")}
            self.total_volume -= oldest_volumes["buy"] + oldest_volumes["sell"]
            self.total_volume += buy_volume + sell_volume
            self.volume_imbalance -= oldest_volumes["buy"] - oldest_volumes["sell"]
            self.volume_imbalance += buy_volume - sell_volume
            self._update_volumes(buy_volume, sell_volume)
            self._update_current_value()

    def _update_current_value(self):
        if self.total_volume != 0:
            self.current_value = self.volume_imbalance / self.total_volume
        else:
            self.current_value = 1 / 2

    def _update_volumes(self, buy_volume: int, sell_volume: int):
        self.volumes["buy"].append(buy_volume)
        self.volumes["sell"].append(sell_volume)


########################################################################################################################
#                                                  Agent features                                                      #
########################################################################################################################


class Inventory(Feature):
    def __init__(
        self,
        name: str = "Inventory",
        min_value: float = -1000000,
        max_value: float = 1000000,
        update_frequency: timedelta = timedelta(seconds=1),
        normalisation_on: bool = False,
    ):
        super().__init__(name, min_value, max_value, update_frequency, 0, normalisation_on)

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        super()._reset(state, first_usage_time)

    def _update(self, state: State) -> None:
        self.current_value = state.portfolio.inventory


class SellDistance(Feature):
    def __init__(
        self,
        name: str = "SellDistance",
        update_frequency: timedelta = timedelta(seconds=1),
        normalisation_on: bool = False,
    ):
        super().__init__(name, 0, 1000000, update_frequency, 0, normalisation_on)

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        super()._reset(state, first_usage_time)

    def _update(self, state: State) -> None:
        if state.sell_parameter:
            self.current_value = state.sell_parameter * state.orderbook.spread/2
        else:
            self.current_value = 0


class BuyDistance(Feature):
    def __init__(
        self,
        name: str = "BuyDistance",
        update_frequency: timedelta = timedelta(seconds=1),
        normalisation_on: bool = False
    ):
        super().__init__(name, 0, 1000000, update_frequency, 0, normalisation_on)

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        super()._reset(state, first_usage_time)

    def _update(self, state: State) -> None:
        if state.buy_parameter:
            self.current_value = state.buy_parameter * state.orderbook.spread/2
        else:
            self.current_value = 0
