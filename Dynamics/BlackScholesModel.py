import numpy as np

from PDESolver2.Common.DynamicsType import DynamicsType
from PDESolver2.Dynamics.Dynamics import Dynamics
from PDESolver2.MarketParameter.DividendRate.DividendRate import DividendRate
from PDESolver2.MarketParameter.InterestRate.InterestRate import InterestRate
from PDESolver2.MarketParameter.Volatility.Volatility import Volatility


class BlackScholsModel(Dynamics):

    def __init__(self, volatility: Volatility, interest_rate: InterestRate, dividend_rate: DividendRate,
                 correlation: float):
        self._volatility: Volatility = volatility
        self._interest_rate: InterestRate = interest_rate
        self._dividend_rate: DividendRate = dividend_rate
        self._will_parameters_change = (self._volatility.will_change() or self._interest_rate.will_change()
                                        or self._dividend_rate.will_change())
        self._current_time: float = 0
        self._discretized_domain: np.array = None
        self._correlation: float = correlation

    def get_dynamics_type(self) -> DynamicsType:
        return DynamicsType.BLACK_SCHOLES

    def get_volatility(self) -> np.array:
        return self._volatility.get_volatility_array(self._discretized_domain, self._current_time)

    def get_interest_rate(self) -> np.array:
        return self._interest_rate.get_interest_rate_array(self._discretized_domain, self._current_time)

    def get_dividend_rate(self) -> np.array:
        return self._dividend_rate.get_dividend_rate_array(self._discretized_domain, self._current_time)

    def null_dividend_rate(self) -> bool:
        return not self._dividend_rate.will_change() and self._dividend_rate.is_null()

    def get_correlation(self) -> float:
        return self._correlation

    def will_parameters_change(self) -> bool:
        return self._will_parameters_change

    def set_current_time(self, new_current_time: float):
        self._current_time = new_current_time

    def set_discretized_domain(self, discretized_domain: np.array):
        self._discretized_domain = discretized_domain
