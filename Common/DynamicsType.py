from enum import Enum, auto


class DynamicsType(Enum):
    BLACK_SCHOLES = "Black-Scholes dynamics"
    LEVY = "Levy dynamics"
