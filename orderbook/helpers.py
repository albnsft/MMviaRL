from itertools import chain

import pandas as pd
import plotly.express as px

from orderbook.models import Orderbook


def visualise_orderbook(
    orderbook: Orderbook, n_levels: int = 5, tick_size: float = 0.01
):
    df = convert_orderbook_to_dataframe(orderbook, n_levels)
    df.price = df.price
    fig = px.bar(
        df,
        x="price",
        y="volume",
        color="direction",
        title=orderbook.ticker,
        color_discrete_sequence=["green", "red"],
    )
    fig.update_traces(width=tick_size)
    fig.show()


def convert_orderbook_to_dataframe(orderbook: Orderbook, n_levels: int = 10):
    order_dict = {}
    for direction in ["buy", "sell"]:
        prices = reversed(getattr(orderbook, direction)) if direction == "buy" else getattr(orderbook, direction)
        for level, price in enumerate(prices):
            if level >= n_levels:
                break
            total_volume = sum(order.volume for order in getattr(orderbook, direction)[price])  # type: ignore
            order_dict[direction + "_" + str(level)] = (direction, price, total_volume)
    df = pd.DataFrame(order_dict).T
    return df.rename(columns={0: "direction", 1: "price", 2: "volume"})


def convert_orderbook_to_series(orderbook: Orderbook, n_levels: int = 10):
    order_dict = {}
    for direction in ["buy", "sell"]:
        prices = reversed(getattr(orderbook, direction)) if direction == "buy" else getattr(orderbook, direction)
        for level, price in enumerate(prices):
            if level >= n_levels:
                break
            total_volume = sum(order.volume for order in getattr(orderbook, direction)[price])  # type: ignore
            order_dict[direction + "_price_" + str(level)] = price
            order_dict[direction + "_volume_" + str(level)] = total_volume
    return pd.DataFrame(order_dict, index=[0])


def get_book_columns(n_levels: int = 5):
    price_cols = list(chain(*[("sell_price_{0},buy_price_{0}".format(i)).split(",") for i in range(n_levels)]))
    volume_cols = list(chain(*[("sell_volume_{0},buy_volume_{0}".format(i)).split(",") for i in range(n_levels)]))
    return list(chain(*zip(price_cols, volume_cols)))


def convert_to_lobster_format(orderbook: Orderbook, n_levels: int = 5):
    lobster_book = dict()
    for direction in ["buy", "sell"]:
        half_book = getattr(orderbook, direction)  # type: ignore
        if direction == "buy":
            half_book = reversed(half_book)
        for level, price in enumerate(half_book):
            if level < n_levels:
                lobster_book[direction + "_price_" + str(level)] = float(price)
                volume = 0
                for order in getattr(orderbook, direction)[price]:  # type: ignore
                    volume += order.volume
                lobster_book[direction + "_volume_" + str(level)] = float(volume)
    return lobster_book


def compare_elements_of_books(book_1: pd.DataFrame, book_2: pd.DataFrame, verbose: bool = False):
    if verbose:
        for key in book_1.keys():
            print(f"{key}: {book_1[key]==book_2[key]}")
    else:
        print(all([book_1[key] == book_2[key] for key in book_1.keys()]))
