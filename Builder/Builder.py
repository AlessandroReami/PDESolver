import math
from typing import List, Optional, Union

from PDESolver2.Common.DiscretizerType import DiscretizerType
from PDESolver2.Common.DynamicsType import DynamicsType
from PDESolver2.Common.FiniteDifferencesType import FiniteDifferencesType

from PDESolver2.Common.OptionType.OptionType import OptionType

from PDESolver2.Common.TimeMethodType import TimeMethodType
from PDESolver2.Discretizer.BarrierOptionDiscretizer import BarrierOptionDiscretizer
from PDESolver2.Discretizer.Discretizer import Discretizer
from PDESolver2.Discretizer.FEMDiscretizer import FEMDiscretizer
from PDESolver2.Dynamics.BlackScholesModel import BlackScholsModel
from PDESolver2.Dynamics.Dynamics import Dynamics
from PDESolver2.MarketParameter.DividendRate.DividendRate import DividendRate
from PDESolver2.MarketParameter.InterestRate.InterestRate import InterestRate
from PDESolver2.MarketParameter.Volatility.Volatility import Volatility
from PDESolver2.Option.Option import Option
from PDESolver2.Discretizer.FiniteDifferencesDiscretizer import FiniteDifferencesDiscretizer


class Builder:

    def __init__(self):
        self._discretizer: Optional[Discretizer] = None
        self._dynamics: Optional[Dynamics] = None
        self._option: Optional[Option] = None

    def build_discretizer(self, discretizer_type: Union[DiscretizerType, FiniteDifferencesType],
                          domain: List[List[float]], time_to_maturity: float, time_steps: int,
                          space_subdivision: List[int], time_method: TimeMethodType, abs_tol: float = 1e-4,
                          rel_tol: float = 1e-4, max_iter: int = 100):
        if self._option is None or self._dynamics is None:
            raise Exception("discretizer can be built only after option and dynamics")
        if discretizer_type.get_macro_discretizer_type() == DiscretizerType.FINITE_DIFFERENCES:

            if self._option.has_barriers():
                self._discretizer = BarrierOptionDiscretizer(self._option, self._dynamics, domain, time_to_maturity,
                                                             time_steps, space_subdivision,
                                                             time_method, discretizer_type, abs_tol, rel_tol,
                                                             max_iter, self._option.get_upper_barrier(),
                                                             self._option.get_lower_barrier())
            else:
                self._discretizer = FiniteDifferencesDiscretizer(self._option, self._dynamics, domain, time_to_maturity,
                                                                 time_steps, space_subdivision,
                                                                 time_method, discretizer_type, abs_tol, rel_tol,
                                                                 max_iter)

        elif discretizer_type.get_macro_discretizer_type() == DiscretizerType.FEM:
            self._discretizer = FEMDiscretizer(self._option, self._dynamics, domain, time_to_maturity, time_steps,
                                               space_subdivision, time_method,
                                               abs_tol, rel_tol, max_iter, self._option.get_lower_barrier(),
                                                             self._option.get_upper_barrier())
        else:
            raise Exception("ERROR.")

        return self._discretizer

    def build_option(self, underlying_values: List[float], strike: List[float],
                     option_type: OptionType, lower_barriers: List[float] = None,
                     upper_barriers: List[float] = None) -> Option:

        lower_barriers = lower_barriers if lower_barriers is not None else [0 for _ in range(len(strike))]
        upper_barriers = upper_barriers if upper_barriers is not None else [math.inf for _ in range(len(strike))]

        if (len(underlying_values) != len(strike) or len(strike) != option_type.get_dimension().value
                or len(lower_barriers) != len(strike) or len(upper_barriers) != len(strike)):
            raise ValueError("Given parameters don't match. Check List length and requested dimension.")

        self._option = Option(strike, option_type, lower_barriers, upper_barriers)

        return self._option

    def build_dynamics(self, dynamics_type: DynamicsType, volatility: Volatility, interest_rate: InterestRate,
                       dividend_rate: DividendRate, correlation: float = 0):
        if dynamics_type == DynamicsType.BLACK_SCHOLES:
            self._dynamics = BlackScholsModel(volatility, interest_rate, dividend_rate, correlation=correlation)
        else:
            raise Exception("Not yet developed")
        return self._dynamics
