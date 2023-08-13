import sys
from collections import deque
from datetime import datetime, timedelta

if sys.version_info[0] == 3 and sys.version_info[1] >= 8:
    from typing import Deque, Dict, List, Optional, Literal, Callable, cast
else:
    from typing import Deque, Dict, List, Optional, Callable
    from typing_extensions import Literal

import numpy as np
import pandas as pd

from database.HistoricalDatabase import HistoricalDatabase
from orderbook.models import Orderbook, Order, LimitOrder, FilledOrders, OrderDict
from orderbook.Exchange import Exchange
from simulation.HistoricalOrderGenerator import HistoricalOrderGenerator


class OrderbookSimulator:
    def __init__(
        self,
        ticker: str = "MSFT",
        exchange: Exchange = None,
        order_generator: HistoricalOrderGenerator = None,
        n_levels: int = 5,
        database: HistoricalDatabase = None,
        outer_levels: int = 5,
        trading_date: datetime = datetime(2012, 6, 21),
        verbose: bool = False
    ) -> None:
        self.ticker = ticker
        self.exchange = exchange or Exchange(ticker)
        self.order_generator = order_generator or HistoricalOrderGenerator(ticker, database)
        self.now_is: datetime = datetime(2000, 1, 1)
        self.trading_date = trading_date
        self.n_levels = n_levels
        self.database = database or HistoricalDatabase()
        self.outer_levels = outer_levels
        self.verbose = verbose
        # The following is for re-syncronisation with the historical data
        self.max_sell_price: int = 0
        self.min_buy_price: int = np.infty  # type:ignore
        self.initial_buy_price_range: int = np.infty  # type:ignore
        self.initial_sell_price_range: int = np.infty  # type:ignore

    def reset_episode(self, start_date: datetime, start_book: Optional[Orderbook] = None):
        if not start_book:
            start_book = self.get_historical_start_book(start_date)
        self.exchange.central_orderbook = start_book
        self._reset_initial_price_ranges()
        assert start_date.microsecond == 0, "Episodes must be started on the second."
        self.now_is = start_date
        return start_book

    def forward_step(self, until: datetime, internal_orders: Optional[List[Order]] = None) -> FilledOrders:
        assert (
            until > self.now_is
        ), f"The current time is {self.now_is.time()}, but we are trying to step forward in time until {until.time()}!"
        external_orders = list(self.order_generator.generate_orders(self.now_is, until))
        orders = internal_orders or list()
        orders += external_orders
        filled_internal_orders = []
        filled_external_orders = []
        self.does_cancel_internal_orders()
        for order in orders:
            filled = self.exchange.process_order(order)
            if filled:
                filled_internal_orders += filled.internal
                filled_external_orders += filled.external
        self.now_is = until
        if (self._near_exiting_initial_price_range or self._exiting_worst_price) :
            self.update_outer_levels()
        return FilledOrders(internal=filled_internal_orders, external=filled_external_orders)

    def get_historical_start_book(self, start_date: datetime) -> Orderbook:
        start_series = self.database.get_last_snapshot(start_date, ticker=self.ticker)
        assert len(start_series) > 0, f"There is no data before the episode start time: {start_date}"
        initial_orders = self._get_initial_orders_from_snapshot(start_series)
        return self.exchange.get_initial_orderbook_from_orders(initial_orders)

    def _initial_prices_filter_function(self, direction: Literal["buy", "ask"], price: int) -> bool:
        if direction == "buy" and price < self.min_buy_price or direction == "sell" and price > self.max_sell_price:
            return True
        else:
            return False

    def update_outer_levels(self) -> None:
        if self.verbose: print(f"Updating outer levels. Current time is {self.now_is}.")
        orderbook_series = self.database.get_last_snapshot(self.now_is, ticker=self.ticker)
        orders_to_add = self._get_initial_orders_from_snapshot(orderbook_series, self._initial_prices_filter_function)
        for order in orders_to_add:
            getattr(self.exchange.central_orderbook, order.direction)[order.price] = deque([order])
        self.min_buy_price = min(self.min_buy_price, self.exchange.orderbook_price_range[0])
        self.max_sell_price = max(self.max_sell_price, self.exchange.orderbook_price_range[1])

    def _cancel_internal_orders(self, idx_cancel, central_orderbook_side):
        prices = list(central_orderbook_side.keys())
        orders_to_cancel = []
        for idx in idx_cancel:
            for order in central_orderbook_side[prices[idx]]:
                if not order.is_external:
                    orders_to_cancel.append(order)
        if len(orders_to_cancel) >= 1: print("some agent orders are no longer competitive --> cancellation")
        for order in orders_to_cancel:
            self.exchange.remove_order(order)

    def does_cancel_internal_orders(self):
        mid_price = self.exchange.central_orderbook.midprice
        idx_buy_cancel = np.where(np.array(self.exchange.central_orderbook.buy.keys()) >= mid_price)[0]
        idx_sell_cancel = np.where(np.array(self.exchange.central_orderbook.sell.keys()) <= mid_price)[0]
        if len(idx_buy_cancel) > 0:
            self._cancel_internal_orders(idx_buy_cancel, self.exchange.central_orderbook.buy)
        if len(idx_sell_cancel) > 0:
            self._cancel_internal_orders(idx_sell_cancel, self.exchange.central_orderbook.sell)

    @staticmethod
    def _remove_hidden_executions(messages: pd.DataFrame):
        return messages[messages.message_type != "execution_hidden"]

    always_true_function: Callable = lambda direction, price: True

    def _get_initial_orders_from_snapshot(self, series: pd.DataFrame, filter_function: Callable = always_true_function):
        initial_orders = []
        for direction in ["buy", "sell"]:
            for level in range(self.n_levels):
                if f"{direction}_volume_{level}" not in series:
                    continue
                if filter_function(direction, series[f"{direction}_price_{level}"]):
                    initial_orders.append(
                        LimitOrder(
                            timestamp=series.name,
                            price=series[f"{direction}_price_{level}"],
                            volume=series[f"{direction}_volume_{level}"],
                            direction=direction,  # type: ignore
                            ticker=self.ticker,
                            internal_id=-1,
                            external_id=None,
                            is_external=True,
                        )
                    )
        return initial_orders

    @property
    def _near_exiting_initial_price_range(self) -> bool:
        outer_proportion = self.outer_levels / self.n_levels
        return (
            self.exchange.best_buy_price < self.min_buy_price + outer_proportion * self.initial_buy_price_range
            or self.exchange.best_sell_price > self.max_sell_price - outer_proportion * self.initial_sell_price_range
        )

    @property
    def _exiting_worst_price(self) -> bool:
        worst_buy, worst_sell = self.exchange.orderbook_price_range
        return (
            worst_buy < self.min_buy_price
            or worst_sell > self.max_sell_price
        )

    def _reset_initial_price_ranges(self):
        self.min_buy_price, self.max_sell_price = self.exchange.orderbook_price_range
        self.initial_buy_price_range = self.exchange.best_buy_price - self.min_buy_price
        self.initial_sell_price_range = self.max_sell_price - self.exchange.best_sell_price

