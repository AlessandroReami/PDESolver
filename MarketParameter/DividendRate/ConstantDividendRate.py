import numpy as np

from PDESolver2.MarketParameter.DividendRate.DividendRate import DividendRate


class ConstantDividendRate(DividendRate):

    def __init__(self, dividend_rate: float):
        self._dividend_rate: float = dividend_rate

    def get_dividend_rate_array(self, discretized_domain: np.array, current_time: float) -> np.array:
        return np.full_like(discretized_domain, self._dividend_rate)

    def will_change(self) -> bool:
        return False

    def is_null(self) -> bool:
        return self._dividend_rate == 0
