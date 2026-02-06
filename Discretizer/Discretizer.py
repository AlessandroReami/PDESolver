from abc import abstractmethod

from PDESolver2.Common.DiscretizerType import DiscretizerType
from PDESolver2.Common.OptionType.PayoffType import PayoffType

from PDESolver2.Option.Option import Option

import math
import time

import matplotlib.pyplot as plt
from typing import List, Optional
from colorama import Fore
import numpy as np
from numpy.linalg import norm
from scipy import sparse
from scipy.sparse import hstack, vstack
from scipy.sparse.linalg import spsolve

from PDESolver2.Common.OptionType.DimensionType import DimensionType
from PDESolver2.Common.OptionType.OptionType import OptionType
from PDESolver2.Common.TimeMethodType import TimeMethodType
from PDESolver2.Dynamics.Dynamics import Dynamics
from PDESolver2.UsefulFunction.UsefulFunction import black_scholes_price, \
    bs_down_out_put_price, bs_up_out_call_price


class Discretizer:
    def __init__(self, option: Option, dynamics: Dynamics, domain: List[List[float]], time_to_maturity: float,
                 time_steps: int,
                 space_subdivision: List[int], time_method: TimeMethodType, abs_tol: float, rel_tol: float,
                 max_iter: int):
        if len(domain) == 0 or len(space_subdivision) == 0:
            raise Exception("Bad variable definition.")

        if len(space_subdivision) != len(domain):
            raise Exception(
                "Dimension doesn't match. Space steps is supposed to contain the space steps for each dimension")

        self._option: Option = option
        self._dynamics: Dynamics = dynamics
        self._space_subdivision: List[int] = space_subdivision.copy()
        self._time_steps: int = time_steps
        self._time_to_maturity: float = time_to_maturity
        self._time_method: TimeMethodType = time_method
        self._current_time: float = 0
        self._current_time_step_counter: int = 0
        self._time_step_size: float = self._time_to_maturity / self._time_steps
        self._abs_tol: float = abs_tol
        self._rel_tol: float = rel_tol
        self._max_iter: int = max_iter
        self._domain: List[List[float]] = domain
        self._discretized_domain: np.array = None
        self.set_domain(domain)
        self._last_time_pde_matrix_update: float = 0

        self._dynamics.set_discretized_domain(self._discretized_domain)
        self._option.compute_payoff(self._discretized_domain)
        self._fair_price: Optional[np.array] = None
        self._discretizer_type: Optional[DiscretizerType] = None
        # self._transparent_condition: bool = not self._dynamics.will_parameters_change() and self._dynamics.null_dividend_rate()
        # if self._transparent_condition:
        #     self._alpha, self._beta, self._gamma, self._delta = self._get_transparent_boundary_condition_coeff()
        #     self._max_memory: int = self._time_steps
        #     self._Pn_memory: List[List[float]] = [[0, 0] for _ in range(self._max_memory)]

    def get_discretizer_type(self):
        pass

    def get_time_step_size(self):
        return self._time_step_size

    def get_dynamics(self):
        return self._dynamics

    def get_domain(self):
        return self._domain

    def get_discretized_domain(self):
        return self._discretized_domain.copy()

    def set_time_step_size(self, new_time_step_size: float):
        self._time_step_size = new_time_step_size

    def get_space_step_size(self, dimension: int):
        return ((self._discretized_domain[dimension][-1] - self._discretized_domain[dimension][0]) /
                self._space_subdivision[dimension])

    def set_domain(self, domain: List[List[float]]):
        discretized_intervals: List[np.ndarray] = []
        reversed_space_subdivision = self._space_subdivision.copy()
        reversed_space_subdivision.reverse()
        for interval in domain:
            if len(interval) != 2:
                raise Exception("Domain is defined with cartesian product of intervals.")
            discretized_intervals.append(np.linspace(interval[0], interval[1], reversed_space_subdivision.pop() + 1))

        discretized_domain_list = self._meshgrid(discretized_intervals)
        self._discretized_domain: np.array = np.array(discretized_domain_list)

    def increase_current_time(self, step: float):
        self._current_time += step
        self._current_time_step_counter += 1

    def get_fair_price(self, recompute: bool = False):
        if self._fair_price is not None and not recompute:
            return self._fair_price
        else:
            Pn = self._option.get_payoff()[0]
            method_type = self._time_method
            if method_type == TimeMethodType.FORWARD_EULER:
                self._fair_price = self._get_forward_euler_fair_value(Pn)
            elif method_type in [TimeMethodType.BACKWARD_EULER, TimeMethodType.CRANK_NICOLSON]:
                self._fair_price = self._get_theta_method_fair_value(method_type)
            elif method_type == TimeMethodType.RUNGE_KUTTA_2_EMBEDDED:
                self._fair_price = self._get_r_k_explicit_embedded_fair_value(method_type)
            elif method_type == TimeMethodType.RUNGE_KUTTA_3_SEMI_IMPL:
                self._fair_price = self._get_r_k_diagonally_impl_fair_value(method_type)
            elif method_type in [TimeMethodType.RUNGE_KUTTA_LOBATTO3B, TimeMethodType.RUNGE_KUTTA_LOBATTO3C]:
                self._fair_price = self._get_r_k_implicit_embedded_fair_value(method_type)
            else:
                raise Exception(f"Not implemented {self._time_method}")
        self._fair_price = self._option.convert_to_fair_price(self._fair_price, self._discretized_domain,
                                                              self._option.get_strike())
        return self._fair_price

    def plot_fair_price(self, path: str, color: str = "blue", label: str = "Option price", plot_payoff: bool = False,
                        plot_analytical: bool= True):
        title_fontsize = 21
        label_fontsize = 19
        figsize = (8.5, 7.5)

        if self._fair_price is None:
            _ = self.get_fair_price()

        if self._option.get_dimension() == DimensionType.DIM1:
            plt.figure(figsize=figsize)
            if self._option.get_payoff_type() in [PayoffType.ASIAN_CALL, PayoffType.ASIAN_PUT]:
                tmp = [(dom, price) for dom, price in zip(self._discretized_domain[0], self._fair_price) if dom>0]
                x = [dom for dom, price in tmp]
                y = [price for dom, price in tmp]
                plt.plot(x, y, color=color, label=label)
            else:
                plt.plot(self._discretized_domain[0], self._fair_price, color=color, label=label)
                self._analytical_prices = [0] * len(self._discretized_domain[0])

                if plot_analytical:
                    if not self._option.has_barriers():
                        self._analytical_prices = black_scholes_price(self._option.get_payoff_type(),
                                                                      self._discretized_domain[0], self._option._strike[0],
                                                                      self._time_to_maturity,
                                                                      self._dynamics.get_interest_rate()[0][0],
                                                                      self._dynamics.get_volatility()[0][0],
                                                                      self._dynamics.get_dividend_rate()[0][0],
                                                                      self._option.get_lower_barrier()[
                                                                          0] if self._option.get_lower_barrier() else None,
                                                                      self._option.get_upper_barrier()[
                                                                          0] if self._option.get_upper_barrier() else None)
                    elif self._option.get_lower_barrier() != [0]:
                        self._analytical_prices = bs_down_out_put_price(self._discretized_domain[0], self._option.get_strike()[0],
                                                                        self._time_to_maturity,
                                                                        self._dynamics.get_interest_rate()[0][0],
                                                                        self._dynamics.get_volatility()[0][0],
                                                                        self._option.get_lower_barrier()[0])
                    else:
                        self._analytical_prices = bs_up_out_call_price(self._discretized_domain[0], self._option.get_strike()[0],
                                                                       self._time_to_maturity,
                                                                       self._dynamics.get_interest_rate()[0][0],
                                                                       self._dynamics.get_volatility()[0][0],
                                                                       self._option.get_upper_barrier()[0])
                    plt.plot(self._discretized_domain[0], self._analytical_prices, color="green", label="Analytical price")

            if plot_payoff:
                plt.plot(self._discretized_domain[0], self._option.get_payoff()[0], color="red", label="Payoff")
            plt.legend(prop={'size': 18})
            plt.xlabel(
                f'Estimated price of an {self._option.get_exercise_type().value} {self._option.get_payoff_type().value} with strike '
                f'{self._option.get_strike()[0]} \n using {self._discretizer_type.value} and {self._time_method.value}',
                fontsize=label_fontsize)
            plt.ylabel('Price', fontsize=label_fontsize)
            plt.title(
                f'Price of an {self._option.get_exercise_type().value} {self._option.get_payoff_type().value}',
                fontsize=title_fontsize)

            plt.savefig(path) if len(path) != 0 else plt.show()
            plt.close()
        else:
            raise Exception(f"Plot not developed for {self._option.get_dimension().name}")

    def plot_pointwise_error(self, path: str, color: str = "blue", label: str = "Pointwise error"):
        title_fontsize = 21
        label_fontsize = 19
        figsize = (8.5, 7.5)

        if self._fair_price is None:
            _ = self.get_fair_price()
        if self._analytical_prices is None:
            raise Exception("Cannot call this method before than plot_fair_price()")

        if self._option.get_dimension() == DimensionType.DIM1:
            plt.figure(figsize=figsize)
            error = [abs(x1 - x2) for x1, x2 in zip(self._fair_price, self._analytical_prices)]
            plt.plot(self._discretized_domain[0], error, color=color, label=label)

            plt.legend(prop={'size': 18})
            plt.xlabel(
                f'Pointwise error with the analyitical solution of an {self._option.get_exercise_type().value} {self._option.get_payoff_type().value} with strike '
                f'{self._option.get_strike()[0]} \n using {self._discretizer_type.value} and {self._time_method.value}',
                fontsize=label_fontsize)
            plt.ylabel('Pointwise Error', fontsize=label_fontsize)
            plt.title(
                f'Price of an {self._option.get_exercise_type().value} {self._option.get_payoff_type().value}',
                fontsize=title_fontsize)

            plt.savefig(path) if len(path) != 0 else plt.show()
            plt.close()
        else:
            raise Exception(f"Plot not developed for {self._option.get_dimension().name}")

    def compute_average_time_needed(self, iterations: int = 25):
        e0 = time.time()
        for i in range(iterations):
            self.get_fair_price(True)
        e1 = time.time()
        print(f"Average execution time: {(e1 - e0) / iterations}")

    @staticmethod
    def _meshgrid(discretized_intervals: List[np.ndarray]):
        x = discretized_intervals[0]
        try:
            y = discretized_intervals[1]
        except IndexError:
            return np.meshgrid(x)
        try:
            z = discretized_intervals[2]
        except IndexError:
            return np.meshgrid(x, y)

        return np.meshgrid(x, y, z)

    @abstractmethod
    def get_matrix(self, option_type: OptionType, dynamics: Dynamics):
        pass

    def _update_pde_matrix(self, current_time: float):
        if self._dynamics.will_parameters_change():
            if self._last_time_pde_matrix_update != current_time:
                self._dynamics.set_current_time(current_time)
                self._pde_matrix = self.get_matrix(self._option.get_option_type(), self._dynamics)
                self._last_time_pde_matrix_update = current_time

    def _get_forward_euler_fair_value(self, Pn):
        k = self._time_step_size
        for i in range(self._time_steps):
            self._update_pde_matrix(self._current_time)
            Pn += k * (self._pde_matrix @ Pn)
            self.increase_current_time(k)
        return Pn

    def _get_theta_method_fair_value(self, method_type: TimeMethodType):
        # y_n+1= y_n + k( (1-theta) * f_n + theta * f_n+1)
        if method_type == TimeMethodType.BACKWARD_EULER:
            theta = 1
        elif method_type == TimeMethodType.CRANK_NICOLSON:
            theta = 1 / 2
        else:
            raise Exception(f" Type: {method_type.name} not recognised")

        Pn = self._option.get_payoff()[0]
        I = sparse.eye(self._pde_matrix.shape[0], format="csr")
        for i in range(self._time_steps):
            # todo si può ottimizzare con una fattorizzazione se la matrice non cambia
            self._update_pde_matrix(self._current_time)

            k = self._time_step_size
            Pn = spsolve(I - theta * k * self._pde_matrix,
                         (I + (1 - theta) * k * self._pde_matrix) @ Pn)
            self.increase_current_time(k)
        return Pn

    def _get_r_k_explicit_embedded_fair_value(self, method_type: TimeMethodType):
        # tableau:    0   |     0
        #            1/2  |    1/2    0
        #            -------------------------
        #                       1      0
        #                       0      1
        if method_type == TimeMethodType.RUNGE_KUTTA_2_EMBEDDED:
            a = [[0, 0], [1 / 2, 0]]
            b_1, b_2 = [1], [0, 1]
            c = [0, 1 / 2]
        else:
            raise Exception(f"Type: {method_type.name} not recognised")
        Pn_rk = self._option.get_payoff()[0]
        iteration_counter = 0
        while self._current_time < self._time_to_maturity and iteration_counter < self._max_iter:
            k = self._time_step_size
            if self._current_time + k > self._time_to_maturity:
                k = self._time_to_maturity - self._current_time

            self._update_pde_matrix(self._current_time)

            Pn_eu_new, Pn_rk_new = self.__one_step_r_k_embedded_espl(Pn_rk, a, b_1, b_2, c, k)
            error = norm(Pn_rk_new - Pn_eu_new, np.inf)
            Pn_eu_norm = norm(Pn_eu_new, np.inf)
            if error <= self._abs_tol + Pn_eu_norm * self._rel_tol:  # approved
                self.increase_current_time(k)
                Pn_rk = Pn_rk_new.copy()
                iteration_counter = -1
            # update of next step size
            k *= min(2, max(0.5, 0.7 * math.sqrt((self._abs_tol + Pn_eu_norm * self._rel_tol) / error)))
            self.set_time_step_size(k)
            iteration_counter += 1
        if self._max_iter == iteration_counter:
            print(f"{Fore.RED}ERROR: Max number of iteration reached!{Fore.RESET}")
        return Pn_rk

    def _get_r_k_implicit_embedded_fair_value(self, method_type: TimeMethodType):
        # tableau 3B:    0   |  1/6  -1/6    0
        #               1/2  |  1/6   1/3    0
        #                1   |  1/6   5/6    0
        #               -------------------------
        #                       1/6   2/3    1/6
        #                      -1/2    2    -1/2
        #
        # tableau 3C:    0   |  1/6  -1/3    1/6
        #               1/2  |  1/6   5/12  -1/12
        #                1   |  1/6   2/3    1/6
        #               -------------------------
        #                       1/6   2/3    1/6
        #                      -1/2    2    -1/2
        if method_type == TimeMethodType.RUNGE_KUTTA_LOBATTO3B:
            a = [[1 / 6, -1 / 6, 0],
                 [1 / 6, 1 / 3, 0],
                 [1 / 6, 5 / 6, 0]]
            b_1 = [1 / 6, 2 / 3, 1 / 6]
            b_2 = [-1 / 2, 2, -1 / 2]
            c = [0, 1 / 2, 1]
        elif method_type == TimeMethodType.RUNGE_KUTTA_LOBATTO3C:
            a = [[1 / 6, -1 / 3, 1 / 6],
                 [1 / 6, 5 / 12, -1 / 12],
                 [1 / 6, 2 / 3, 1 / 6]]
            b_1 = [1 / 6, 2 / 3, 1 / 6]
            b_2 = [-1 / 2, 2, -1 / 2]
            c = [0, 1 / 2, 1]
        else:
            raise Exception(f"Type: {method_type.name} not recognised")
        Pn_ord4 = self._option.get_payoff()[0]
        iteration_counter = 0
        while self._current_time < self._time_to_maturity and iteration_counter < self._max_iter:
            k = self._time_step_size
            if self._current_time + k > self._time_to_maturity:
                k = self._time_to_maturity - self._current_time

            self._update_pde_matrix(self._current_time)

            Pn_ord3_new, Pn_ord4_new = self.__one_step_r_k_embedded_impl(Pn_ord4, a, b_1, b_2, c, k)
            error = norm(Pn_ord4_new - Pn_ord3_new, np.inf)
            Pn_ord3_norm = norm(Pn_ord3_new, np.inf)
            if error <= self._abs_tol + Pn_ord3_norm * self._rel_tol:  # approved
                self.increase_current_time(k)
                Pn_ord4 = Pn_ord4_new.copy()
                iteration_counter = -1
            # update of next step size
            k *= min(2, max(0.5, 0.7 * math.sqrt((self._abs_tol + Pn_ord3_norm * self._rel_tol) / error)))
            self.set_time_step_size(k)
            iteration_counter += 1
        if self._max_iter == iteration_counter:
            print(f"{Fore.RED}Max number of iteration reached {Fore.RESET}")
        return Pn_ord4

    def _get_r_k_diagonally_impl_fair_value(self, method_type: TimeMethodType):
        # tableau:   (3+sqrt(3))/6  | (3+sqrt(3))/6
        #            (3-sqrt(3))/6  |  - sqrt(3)/3      (3+sqrt(3))/6
        #            ----------------------------------------------------
        #                           |      1/2               1/2
        if method_type == TimeMethodType.RUNGE_KUTTA_3_SEMI_IMPL:
            a = [[(3 + np.sqrt(3)) / 6, 0],
                 [- np.sqrt(3) / 3, (3 + np.sqrt(3)) / 6]]
            b = [1 / 2, 1 / 2]
            c = [(3 + np.sqrt(3)) / 6, (3 - np.sqrt(3)) / 6]  # todo se f dipende da t, le formule sotto vanno cambiate
        else:
            raise Exception(f"Type: {method_type.name} not recognised")
        Pn = self._option.get_payoff()[0]
        for i in range(self._time_steps):
            k = self._time_step_size
            self._update_pde_matrix(self._current_time)
            Pn = self.__one_step_r_k_semi_impl(Pn, a, b, c, k)
            self.increase_current_time(k)
        return Pn

    def __one_step_r_k_semi_impl(self, Pn: np.array, a: List[List[float]], b: List[float], c: List[float], k: float):
        # todo considera c
        xi = np.zeros((len(Pn), len(b)))
        I = sparse.eye(self._pde_matrix.shape[0], format="csr")

        for i in range(len(b)):
            xi[:, i] = spsolve(I - k * a[i][i] * self._pde_matrix,
                               Pn + k * (sum(a[i][j] * self._pde_matrix @ xi[:, j] for j in range(i))))
        return Pn + k * (sum(b[i] * self._pde_matrix @ xi[:, i] for i in range(len(b))))

    def __one_step_r_k_embedded_espl(self, Pn: np.array, a: List[List[float]], b_1: List[float], b_2: List[float],
                                     c: List[float], k: float):
        # todo introduci c
        xi = np.zeros((len(Pn), len(b_2)))
        for i in range(len(b_2)):
            xi[:, i] = Pn + k * (sum(a[i][j] * self._pde_matrix @ xi[:, j] for j in range(i)))
        return (Pn + k * (sum(b_1[i] * self._pde_matrix @ xi[:, i] for i in range(len(b_1)))),
                Pn + k * (sum(b_2[i] * self._pde_matrix @ xi[:, i] for i in range(len(b_2)))))

    def __one_step_r_k_embedded_impl(self, Pn: np.array, a: List[List[float]], b_1: List[float], b_2: List[float],
                                     c: List[float], k: float):
        # todo introduci c
        Pn_concat = np.concatenate(([Pn] * len(b_2)))
        matrix = vstack(
            [hstack([a[i][j] * self._pde_matrix for j in range(len(b_2))], format="csr") for i in range(len(b_2))])
        I = sparse.eye(matrix.shape[0], format="csr")
        xi = spsolve(I - k * matrix, Pn_concat)
        return (Pn + k * (sum(b_1[i] * self._pde_matrix @ xi[i * len(Pn): (i + 1) * len(Pn)] for i in range(len(b_1)))),
                Pn + k * (sum(b_2[i] * self._pde_matrix @ xi[i * len(Pn): (i + 1) * len(Pn)] for i in range(len(b_2)))))

    # def _get_transparent_boundary_condition_coeff(self) -> tuple[float, float, float, float]:
    #     # can be used only for constant r, sigma
    #     # q can be added but for the moment it's not developed
    #
    #     delta_t = self._time_step_size
    #     r = self._dynamics.get_interest_rate()[0][0]
    #     sigma = self._dynamics.get_volatility()[0][0]
    #     S = self._domain[0][-1]
    #     h = self.get_space_step_size(0)
    #
    #     a = math.sqrt(2 * math.pi) + math.sqrt(delta_t) * ((r - sigma ** 2 / 2) / sigma + 2 * sigma * S / h)
    #     b = 2 * sigma * S * math.sqrt(delta_t) / h
    #     c = -math.sqrt(delta_t) * ((r - sigma ** 2 / 2) / (2 * sigma) + sigma * S / h)
    #     d = -sigma * S * math.sqrt(delta_t) / h
    #
    #     return a, b, c, d
