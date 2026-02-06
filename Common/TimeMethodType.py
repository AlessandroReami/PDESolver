from enum import Enum


class TimeMethodType(Enum):
    FORWARD_EULER = "Forward_Euler"
    BACKWARD_EULER = "Backward_Euler"
    CRANK_NICOLSON = "Crank_Nicolson"
    RUNGE_KUTTA_2_EMBEDDED = "Explict_Embedded_Runge_Kutta_2"
    RUNGE_KUTTA_3_SEMI_IMPL = "Diagonally_Implicit_Runge_Kutta_3"
    RUNGE_KUTTA_LOBATTO3C = "Embedded_Runge_Kutta_Lobatto_3C"
    RUNGE_KUTTA_LOBATTO3B = "Embedded_Runge_Kutta_Lobatto_3B"

