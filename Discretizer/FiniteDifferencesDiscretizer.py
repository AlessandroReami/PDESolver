from typing import List
import numpy as np
from scipy import sparse
from scipy.sparse import diags
from PDESolver2.Common.DynamicsType import DynamicsType
from PDESolver2.Common.FiniteDifferencesType import FiniteDifferencesType
from PDESolver2.Common.OptionType.ExerciseType import ExerciseType
from PDESolver2.Common.OptionType.OptionType import OptionType
from PDESolver2.Common.OptionType.PayoffType import PayoffType
from PDESolver2.Common.TimeMethodType import TimeMethodType
from PDESolver2.Discretizer.Discretizer import Discretizer
from PDESolver2.Dynamics.Dynamics import Dynamics
from PDESolver2.Option.Option import Option
from PDESolver2.UsefulFunction.UsefulFunction import element_by_row_mult


class FiniteDifferencesDiscretizer(Discretizer):
    def __init__(self, option: Option, dynamics: Dynamics, domain: List[List[float]], time_to_maturity: float,
                 time_steps: int, space_subdivision: List[int], time_method: TimeMethodType,
                 finite_differences_type: FiniteDifferencesType, abs_tol: float, rel_tol: float, max_iter: int):
        if len(domain) != 1:
            raise Exception("Finite difference is supported only in 1 dimension.")
        super().__init__(option, dynamics, domain, time_to_maturity, time_steps, space_subdivision, time_method,
                         abs_tol, rel_tol, max_iter)
        self._discretizer_type: FiniteDifferencesType = finite_differences_type
        self._pde_matrix = self.get_matrix(self._option.get_option_type(), self._dynamics)

    def get_discretizer_type(self):
        return self._discretizer_type

    def get_matrix(self, option_type: OptionType, dynamics: Dynamics):
        s = self._discretized_domain[0]
        vol = dynamics.get_volatility()
        r = dynamics.get_interest_rate()
        q = dynamics.get_dividend_rate()
        T = self._time_to_maturity
        m = self._space_subdivision[0] + 1

        if self._discretizer_type in [FiniteDifferencesType.ORDER2, FiniteDifferencesType.ORDER4]:
            if option_type.get_dimension().value != 1:
                raise Exception("Finite differences developed only for the 1-dimensional case")
            A, B, I = self.get_f_d_matrices(self._discretizer_type)
            if dynamics.get_dynamics_type() == DynamicsType.BLACK_SCHOLES:
                if (option_type.get_exercise_type() == ExerciseType.EUROPEAN and option_type.get_payoff_type() in
                        [PayoffType.CALL, PayoffType.PUT, PayoffType.DIGITAL_CALL, PayoffType.DIGITAL_PUT]):
                    matrix = (element_by_row_mult(0.5 * np.power(vol * s, 2), A) + element_by_row_mult((r - q) * s, B)
                              - element_by_row_mult(r, I))
                elif option_type.get_exercise_type() == ExerciseType.EUROPEAN and option_type.get_payoff_type() in [
                    PayoffType.ASIAN_CALL, PayoffType.ASIAN_PUT]:
                    # formula semplificata per il solo caso 1 dimensionale
                    xi = s
                    matrix = element_by_row_mult(0.5 * np.power(vol * xi, 2), A) - element_by_row_mult(1 / T + xi * r,
                                                                                                       B)
                else:
                    raise Exception("Not developed")
            # elif dynamics.get_dynamics_type()== DynamicsType.LEVY:
            # troppo difficile per ora
            else:
                raise Exception(
                    f"Combination {option_type.get_payoff_type()}, {option_type.get_exercise_type()} not developed.")

            return matrix
        else:
            raise Exception(f"Not implemented {self._discretizer_type}")

    def get_f_d_matrices(self, f_d_type: FiniteDifferencesType):
        # pag 27 e 58
        h = self.get_space_step_size(0)
        m = self._space_subdivision[0] + 1
        if f_d_type == FiniteDifferencesType.ORDER2:
            A = (1 / h ** 2) * diags([1, -2, 1], offsets=[-1, 0, 1], shape=(m, m), format="csr")
            B = (1 / h) * diags([-1 / 2, 1 / 2], offsets=[-1, 1], shape=(m, m), format="csr")
            correction_A, correction_B = [2, -5, 4, -1], [-3 / 2, 2, -1 / 2]
            A[0, 0:4] = (1 / h ** 2) * np.array(correction_A)
            B[0, 0:3] = (1 / h) * np.array(correction_B)
            correction_A.reverse()
            correction_B.reverse()
            A[-1, m - 4:m] = (1 / h ** 2) * np.array(correction_A)
            B[-1, m - 3:m] = (1 / h) * np.array([-x for x in correction_B])
        elif f_d_type == FiniteDifferencesType.ORDER4:
            A = (1 / h ** 2) * diags([-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12], offsets=[-2, -1, 0, 1, 2], shape=(m, m),
                                     format="csr")
            B = (1 / h) * diags([1 / 12, -2 / 3, 2 / 3, -1 / 12], offsets=[-2, -1, 1, 2], shape=(m, m), format="csr")
            A_first_row = [15 / 4, -77 / 6, 107 / 6, -13, 61 / 12, -5 / 6]
            A_second_row = [10 / 12, -10 / 8, -2 / 6, 14 / 12, -2 / 4, 2 / 24]

            B_first_row = [-25 / 12, 4, -3, 4 / 3, -1 / 4]
            B_second_row = [-1 / 4, -5 / 6, 3 / 2, -1 / 2, 1 / 12]

            A[0, 0:6], A[1, 0:6] = (1 / h ** 2) * np.array(A_first_row), (1 / h ** 2) * np.array(A_second_row)
            A_first_row.reverse(), A_second_row.reverse()
            A[-1, m - 6:m], A[-2, m - 6:m] = (1 / h ** 2) * np.array(A_first_row), (1 / h ** 2) * np.array(A_second_row)

            B[0, 0:5], B[1, 0:5] = (1 / h) * np.array(B_first_row), (1 / h) * np.array(B_second_row)
            B_first_row.reverse(), B_second_row.reverse()
            B[-1, m - 5:m], B[-2, m - 5:m] = (1 / h) * np.array([-x for x in B_first_row]), (1 / h) * np.array(
                [-x for x in B_second_row])
        else:
            raise Exception(f"Not implemented {f_d_type}")

        I = sparse.eye(m, format="csr")
        return A, B, I
