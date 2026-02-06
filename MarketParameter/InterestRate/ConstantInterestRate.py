import numpy as np

from PDESolver2.MarketParameter.InterestRate.InterestRate import InterestRate


class ConstantInterestRate(InterestRate):

    def __init__(self, interest_rate: float):
        self._interest_rate: float = interest_rate

    def get_interest_rate_array(self, discretized_domain: np.array, current_time: float) -> np.array:
        return np.full_like(discretized_domain, self._interest_rate)

    def will_change(self) -> bool:
        return False
