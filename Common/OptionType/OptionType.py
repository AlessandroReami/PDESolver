import numpy as np

from PDESolver2.Common.OptionType.DimensionType import DimensionType
from PDESolver2.Common.OptionType.ExerciseType import ExerciseType
from PDESolver2.Common.OptionType.PayoffType import PayoffType


class OptionType:

    def __init__(self, dimension_type: DimensionType, exercise_type: ExerciseType, payoff_type: PayoffType):
        self._dimension_type: DimensionType = dimension_type
        self._exercise_type: ExerciseType = exercise_type
        self._payoff_type: PayoffType = payoff_type

    def get_dimension(self) -> DimensionType:
        return self._dimension_type

    def get_exercise_type(self) -> ExerciseType:
        return self._exercise_type

    def get_payoff_type(self) -> PayoffType:
        return self._payoff_type

    def compute_payoff(self, discretized_domain: np.array, strike: np.array):
        if self._dimension_type.value != 1:
            pass
        elif self._dimension_type.value == 1 and self._payoff_type == PayoffType.CALL and self._exercise_type == ExerciseType.EUROPEAN:
            return np.array(np.maximum(discretized_domain - strike[0], 0))
        elif self._dimension_type.value == 1 and self._payoff_type == PayoffType.PUT and self._exercise_type == ExerciseType.EUROPEAN:
            return np.array(np.maximum(strike[0] - discretized_domain, 0))
        elif self._dimension_type.value == 1 and self._payoff_type == PayoffType.DIGITAL_CALL and self._exercise_type == ExerciseType.EUROPEAN:
            return np.array([[1 if x >= strike[0] else 0 for x in discretized_domain[0]]])
        elif self._dimension_type.value == 1 and self._payoff_type == PayoffType.DIGITAL_PUT and self._exercise_type == ExerciseType.EUROPEAN:
            return np.array([[0 if x >= strike[0] else 1 for x in discretized_domain[0]]])
        elif self._dimension_type.value == 1 and self._payoff_type == PayoffType.ASIAN_CALL and self._exercise_type == ExerciseType.EUROPEAN:
            return np.array([[-xi if xi <= 0 else 0 for xi in discretized_domain[0]]])
        elif self._dimension_type.value == 1 and self._payoff_type == PayoffType.ASIAN_PUT and self._exercise_type == ExerciseType.EUROPEAN:
            return np.array([[0 if xi <= 0 else xi for xi in discretized_domain[0]]])
        else:
            raise Exception("Not developed")

    def convert_to_fair_price(self, fair_price: np.array, discretized_domain: np.array, strike: np.array):
        if self._dimension_type.value == 1 and self._payoff_type == PayoffType.CALL and self._exercise_type == ExerciseType.EUROPEAN:
            return fair_price
        elif self._dimension_type.value == 1 and self._payoff_type == PayoffType.PUT and self._exercise_type == ExerciseType.EUROPEAN:
            return fair_price
        elif self._dimension_type.value == 1 and self._payoff_type == PayoffType.DIGITAL_CALL and self._exercise_type == ExerciseType.EUROPEAN:
            return fair_price
        elif self._dimension_type.value == 1 and self._payoff_type == PayoffType.DIGITAL_PUT and self._exercise_type == ExerciseType.EUROPEAN:
            return fair_price
        elif self._dimension_type.value == 1 and self._payoff_type == PayoffType.ASIAN_CALL and self._exercise_type == ExerciseType.EUROPEAN:
            return [el if el >= 0 else 0 for el in (discretized_domain[0] * fair_price)]
        elif self._dimension_type.value == 1 and self._payoff_type == PayoffType.ASIAN_PUT and self._exercise_type == ExerciseType.EUROPEAN:
            return [el if el >= 0 else 0 for el in (discretized_domain[0] * fair_price)]
        else:
            raise Exception("Not developed")
