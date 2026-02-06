import math
from typing import List

import numpy as np
from PDESolver2.Common.FiniteDifferencesType import FiniteDifferencesType
from PDESolver2.Common.OptionType.OptionType import OptionType
from PDESolver2.Common.TimeMethodType import TimeMethodType
from PDESolver2.Discretizer.FiniteDifferencesDiscretizer import FiniteDifferencesDiscretizer
from PDESolver2.Dynamics.Dynamics import Dynamics
from PDESolver2.Option.Option import Option


class BarrierOptionDiscretizer(FiniteDifferencesDiscretizer):
    def __init__(self, option: Option, dynamics: Dynamics, domain: List[List[float]], time_to_maturity: float,
                 time_steps: int, space_subdivision: List[int], time_method: TimeMethodType,
                 finite_differences_type: FiniteDifferencesType, abs_tol: float, rel_tol: float, max_iter: int,
                 upper_barriers: np.array, lower_barriers: np.array):
        super().__init__(option, dynamics, domain, time_to_maturity, time_steps, space_subdivision, time_method,
                         finite_differences_type, abs_tol, rel_tol, max_iter)
        self._upper_barrier = upper_barriers
        self._lower_barrier = lower_barriers

        domain = self._discretized_domain
        for i in range(len(domain)):
            if ((not math.isinf(upper_barriers[i]) and domain[i][-1] < upper_barriers[i]) or
                    (lower_barriers[i] != 0 and domain[i][0] > lower_barriers[i])):
                raise Exception("Defined domain is too small, it doesn't reach the barriers. "
                                "Impossible to compute a correct price.")
            if upper_barriers[i] <= lower_barriers[i]:
                raise Exception("Barriers with wrong values.")
        
        self._correct_domain()
        self._correct_pde_matrix()

    def get_matrix(self, option_type: OptionType, dynamics: Dynamics):
        return super().get_matrix(option_type, dynamics)
    
    def _correct_pde_matrix(self):
        for i in range(len(self._lower_barrier)):
            if self._discretized_domain[i][0] == self._lower_barrier[i]:
                self._pde_matrix[0, :] = 0
            elif self._lower_barrier[i] != 0:
                raise Exception("Hai fatto del casino con la lower barrier")
            if self._discretized_domain[i][-1] == self._upper_barrier[i]:
                self._pde_matrix[-1, :] = 0
            elif not math.isinf(self._upper_barrier[i]):
                raise Exception("Hai fatto del casino con la upper barrier")

    def _correct_domain(self):
        # si deve usare un dominio ridotto perchè fuori è 0
        new_domain = []
        for i in range(len(self._discretized_domain)):
            new_domain.append([max(self._discretized_domain[i][0], self._lower_barrier[i]),
                               min(self._discretized_domain[i][-1], self._upper_barrier[i])])
        self.set_domain(new_domain)
        self._option.compute_payoff(self._discretized_domain)
        for i in range(len(self._option.get_payoff())):
            if self._discretized_domain[i][0] == self._lower_barrier[i]:
                self._option._payoff[i][0] = 0
            if self._discretized_domain[i][-1] == self._upper_barrier[i]:
                self._option._payoff[i][-1] = 0

    def _update_pde_matrix(self, current_time: float):
        super()._update_pde_matrix(current_time)
        self._correct_pde_matrix()
