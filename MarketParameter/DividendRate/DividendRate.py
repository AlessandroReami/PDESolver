from abc import ABC, abstractmethod

import numpy as np


class DividendRate(ABC):
    """
    Dividend rate interface.
    """

    @abstractmethod
    def get_dividend_rate_array(self, discretized_domain: np.array, current_time: float) -> np.array:
        """
        Getter for scalar dividend rate.
        """
        pass

    def will_change(self) -> bool:
        """
        Returns True it the dividend rate is time-dependent, False otherwise
        """
        pass

    def is_null(self) -> bool:
        """
        Returns True it the dividend rate is null, False otherwise
        """
        pass
