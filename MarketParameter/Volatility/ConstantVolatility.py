import numpy as np

from PDESolver2.MarketParameter.Volatility.Volatility import Volatility


class ConstantVolatility(Volatility):

    def __init__(self, volatility: float):
        self._volatility: float = volatility

    def get_volatility_array(self, discretized_domain: np.array, current_time: float) -> np.array:
        return np.full_like(discretized_domain, self._volatility)

    def will_change(self) -> bool:
        return False
