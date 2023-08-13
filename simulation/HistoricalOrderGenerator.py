from collections import deque
from datetime import datetime
from typing import Deque
import warnings

import pandas as pd

from database.HistoricalDatabase import HistoricalDatabase
from orderbook.create_order import create_order
from orderbook.models import Order


class HistoricalOrderGenerator:
    name = "historical"

    def __init__(
        self,
        ticker: str = "MSFT",
        database: HistoricalDatabase = None
    ):
        self.ticker = ticker
        self.database = database or HistoricalDatabase()
        self.exchange_name = "NASDAQ"  # Here, we are only using LOBSTER data for now

    def generate_orders(self, start_date: datetime, end_date: datetime) -> Deque[Order]:
        messages = self.database.get_messages(start_date, end_date, self.ticker)
        messages = self._process_messages_and_add_internal(messages)
        if len(messages) == 0:
            return deque()
        else:
            return deque(messages.internal_message)

    @staticmethod
    def _remove_hidden_executions(messages: pd.DataFrame):
        if messages.empty:
            warnings.warn("DataFrame is empty.")
            return messages
        else:
            assert (
                "cross_trade" not in messages.message_type.unique()
            ), "Trying to step forward before initial cross-trade!"
            return messages[messages.message_type != "market_hidden"]

    @staticmethod
    def _get_mid_datetime(datetime_1: datetime, datetime_2: datetime):
        return (max(datetime_1, datetime_2) - min(datetime_1, datetime_2)) / 2 + min(datetime_1, datetime_2)

    def _process_messages_and_add_internal(self, messages: pd.DataFrame):
        messages = self._remove_hidden_executions(messages)  #
        internal_messages = messages.apply(get_order_from_external_message, axis=1).values
        if len(internal_messages) > 0:
            messages = messages.assign(internal_message=internal_messages)
        return messages


def get_order_from_external_message(message: pd.Series):
    return create_order(
        order_type=message.message_type,
        order_dict=dict(
            timestamp=message.timestamp,
            price=message.price,
            volume=message.volume,
            direction=message.direction,
            ticker=message.ticker,
            internal_id=None,
            external_id=message.external_id,
            is_external=True,
        ),
    )
