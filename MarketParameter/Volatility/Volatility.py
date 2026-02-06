from abc import ABC, abstractmethod
import numpy as np


class Volatility(ABC):
    """
    Volatility interface.
    """

    @abstractmethod
    def get_volatility_array(self, discretized_domain: np.array, current_time: float) -> np.array:
        """
        Getter for scalar volatility.
        """
        pass

    def will_change(self) -> bool:
        """
        Returns True it the volatility is time-dependent, False otherwise
        """
        pass
