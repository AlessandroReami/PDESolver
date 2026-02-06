
import math
from typing import List, Optional
import numpy as np
from PDESolver2.Common.OptionType.OptionType import OptionType


class Option:
    """
    Notice that time t is the time to maturity, so for t=0 we are at maturity and for t=T we are T time before maturity
    """

    def __init__(self, strike: List[float], option_type: OptionType, lower_barriers: List[float],
                 upper_barriers: List[float]):
        self._option_type: OptionType = option_type
        self._strike: np.array = np.array(strike)
        self._payoff: Optional[np.array] = None
        self._last_time_pde_matrix_update: float = 0
        self._fair_price: np.array = None
        self._lower_barrier = lower_barriers
        self._upper_barrier = upper_barriers

    def get_lower_barrier(self):
        return self._lower_barrier

    def get_upper_barrier(self):
        return self._upper_barrier

    def has_barriers(self):
        return (any([barrier != math.inf for barrier in self._upper_barrier]) or
                any([barrier != 0 for barrier in self._lower_barrier]))

    def get_option_type(self):
        return self._option_type

    def get_payoff(self):
        return self._payoff.copy()

    def get_strike(self):
        return self._strike

    def compute_payoff(self, discretized_domain: np.array) -> None:
        self._payoff: np.array = self._option_type.compute_payoff(discretized_domain, self._strike)

    def convert_to_fair_price(self, fair_price: np.array, discretized_domain: np.array, strike: np.array):
        return self._option_type.convert_to_fair_price(fair_price, discretized_domain, strike)

    def get_dimension(self):
        return self._option_type.get_dimension()

    def get_exercise_type(self):
        return self._option_type.get_exercise_type()

    def get_payoff_type(self):
        return self._option_type.get_payoff_type()

