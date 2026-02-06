from enum import Enum
from PDESolver2.Common.DiscretizerType import DiscretizerType


class FiniteDifferencesType(Enum):
    ORDER2 = "Order 2 Finite differences"
    ORDER4 = "Order 4 Finite differences"

    @staticmethod
    def get_macro_discretizer_type() -> DiscretizerType:
        return DiscretizerType.FINITE_DIFFERENCES
