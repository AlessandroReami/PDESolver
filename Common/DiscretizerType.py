from enum import Enum, auto


class DiscretizerType(Enum):
    FINITE_DIFFERENCES = "Finite differences"
    FEM = "Finite elements"

    def get_macro_discretizer_type(self):
        return self

