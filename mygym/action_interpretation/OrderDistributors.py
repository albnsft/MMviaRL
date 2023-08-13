from orderbook.models import Orderbook


class OrderDistributor:
    def __init__(self, volume: int = 100):
        self.volume = volume

    @property
    def limit_orders(self):
        return {0: (1, 1), 1: (2, 2), 2: (3, 3), 3: (4, 4), 4: (5, 5), 5: (1, 3), 6: (3, 1), 7: (2, 5),
                        8: (5, 2)}

    @staticmethod
    def distance_price(tetha_sell: int, tetha_buy: int, spread: float):
        distance_sell = tetha_sell * spread/2
        distance_buy = tetha_buy * spread/2
        return distance_sell, distance_buy

    def pricing_strat(self, tetha_sell: int, tetha_buy: int, spread: float, mid_price: float):
        distance_sell, distance_buy = self.distance_price(tetha_sell, tetha_buy, spread)
        price_sell = mid_price + distance_sell
        price_buy = mid_price - distance_buy
        return round(price_sell, 2), round(price_buy, 2)

    def convert_action(self, action: int = None, orderbook: Orderbook = None):
        assert action in list(range(len(self.limit_orders)))
        tetha_sell, tetha_buy = self.limit_orders.get(action)
        spread = orderbook.spread
        midprice = orderbook.midprice
        price_sell, price_buy = self.pricing_strat(tetha_sell, tetha_buy, spread, midprice)
        return tetha_sell, tetha_buy, {"buy": price_buy, "sell": price_sell}

