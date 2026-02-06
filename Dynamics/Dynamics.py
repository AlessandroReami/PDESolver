from abc import ABC, abstractmethod

import numpy as np

from PDESolver2.Common.DynamicsType import DynamicsType


class Dynamics(ABC):
    """
    Interface for all the dynamics
    """

    @abstractmethod
    def get_dynamics_type(self) -> DynamicsType:
        pass

    @abstractmethod
    def get_volatility(self) -> np.array:
        pass

    @abstractmethod
    def get_interest_rate(self) -> np.array:
        pass

    @abstractmethod
    def get_dividend_rate(self) -> np.array:
        pass

    @abstractmethod
    def null_dividend_rate(self) -> bool:
        pass

    @abstractmethod
    def get_correlation(self) -> float:
        pass

    @abstractmethod
    def will_parameters_change(self) -> bool:
        """
        Returns True if the parameters are time-dependent
        """
        pass

    @abstractmethod
    def set_current_time(self, new_current_time: float):
        pass

    @abstractmethod
    def set_discretized_domain(self, discretized_domain: np.array):
        pass
