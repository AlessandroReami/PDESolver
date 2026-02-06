from abc import ABC, abstractmethod

import numpy as np


class InterestRate(ABC):
    """
    Interest rate interface.
    """

    @abstractmethod
    def get_interest_rate_array(self, discretized_domain: np.array, current_time: float) -> np.array:
        """
        Getter for scalar interest rate.
        """
        pass

    def will_change(self) -> bool:
        """
        Returns True it the interest rate is time-dependent, False otherwise
        """
        pass
