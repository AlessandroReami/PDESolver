import time

import numpy as np

from PDESolver2.Builder.Builder import Builder
from PDESolver2.Common.DynamicsType import DynamicsType
from PDESolver2.Common.FiniteDifferencesType import FiniteDifferencesType
from PDESolver2.Common.OptionType.DimensionType import DimensionType
from PDESolver2.Common.OptionType.ExerciseType import ExerciseType
from PDESolver2.Common.OptionType.OptionType import OptionType
from PDESolver2.Common.OptionType.PayoffType import PayoffType
from PDESolver2.Common.TimeMethodType import TimeMethodType
from PDESolver2.MarketParameter.DividendRate.ConstantDividendRate import ConstantDividendRate
from PDESolver2.MarketParameter.InterestRate.ConstantInterestRate import ConstantInterestRate
from PDESolver2.MarketParameter.Volatility.ConstantVolatility import ConstantVolatility
from PDESolver2.UsefulFunction.UsefulFunction import bs_down_out_put_price, bs_up_out_call_price

test_array_1 = [
    (50, np.infty, PayoffType.DIGITAL_PUT),
    # ricorda che per questi test devi commentare il plot della sol analitica e il calcolo dell'errore
    (70, np.infty, PayoffType.DIGITAL_PUT),

    (0, 150, PayoffType.DIGITAL_CALL),
    (0, 120, PayoffType.DIGITAL_CALL),

    (50, 80, PayoffType.PUT),
    (120, 150, PayoffType.CALL)]
test_array_0 = [
    (0, 110, PayoffType.CALL),
    (0, 120, PayoffType.CALL),
    (0, 130, PayoffType.CALL),
    (0, 140, PayoffType.CALL),
    (0, 150, PayoffType.CALL),
    (0, 160, PayoffType.CALL),
    (0, 170, PayoffType.CALL),
    (0, 180, PayoffType.CALL),
    (0, 190, PayoffType.CALL),

    (10, np.infty, PayoffType.PUT),
    (20, np.infty, PayoffType.PUT),
    (30, np.infty, PayoffType.PUT),
    (40, np.infty, PayoffType.PUT),
    (50, np.infty, PayoffType.PUT),
    (60, np.infty, PayoffType.PUT),
    (70, np.infty, PayoffType.PUT),
    (80, np.infty, PayoffType.PUT),
    (90, np.infty, PayoffType.PUT)]


def test_simple_barrier_option(x: int):
    if x == 0:
        test_array = test_array_0
    else:
        test_array = test_array_1

    for lb, ub, p in test_array:
        start = time.time()
        builder = Builder()
        print(f"{p.value} option, lower barrier:  {lb} upper_barrier {ub}")
        volatility = ConstantVolatility(0.2)
        interest_rate = ConstantInterestRate(0.03)
        dividend_rate = ConstantDividendRate(0)
        correlation = 0.3
        dynamics_type = DynamicsType.BLACK_SCHOLES
        dynamics = builder.build_dynamics(dynamics_type, volatility, interest_rate, dividend_rate,
                                          correlation=correlation)

        underlying_values = [100]
        strike = [100]
        option_type = OptionType(DimensionType.DIM1, ExerciseType.EUROPEAN, p)
        lower_barriers = [lb]
        upper_barriers = [ub]
        _ = builder.build_option(underlying_values, strike, option_type, lower_barriers, upper_barriers)

        discretizer_type = FiniteDifferencesType.ORDER2
        domain = [[lb, 200]] if lb != 0 else [[0.01, ub]]
        # domain = [[lb, ub]]
        time_to_maturity = 0.5
        time_steps = 1000
        space_subdivision = [500]
        time_method_type = TimeMethodType.CRANK_NICOLSON

        discretizer = builder.build_discretizer(discretizer_type, domain, time_to_maturity, time_steps,
                                                space_subdivision,
                                                time_method_type, abs_tol=1e-5, rel_tol=1e-5, max_iter=100)
        fair_price = discretizer.get_fair_price()
        end = time.time()
        path = (f"Test\\TestBarrierOption\\{option_type.get_payoff_type().value}\\{option_type.get_payoff_type().value}"
                f"_upper_barrier_{ub}_lower_barrier_{lb}.png")
        discretizer.plot_fair_price(path=path, color="blue", plot_payoff=True, plot_analytical=(x == 0))
        if x == 0:

            path = (
                f"Test\\TestBarrierOption\\{option_type.get_payoff_type().value}\\PW_error\\{option_type.get_payoff_type().value}"
                f"_upper_barrier_{ub}_lower_barrier_{lb}.png")
            discretizer.plot_pointwise_error(path=path, color="blue")
            S = discretizer.get_discretized_domain()[0]
            K = strike[0]
            T = time_to_maturity
            r = dynamics.get_interest_rate()[0][0]
            q = dynamics.get_dividend_rate()[0][0]
            sigma = dynamics.get_volatility()[0][0]

            if p == PayoffType.PUT:
                analytical_call_option_price = bs_down_out_put_price(S, K, T, r, sigma, lb)
            elif p == PayoffType.CALL:
                analytical_call_option_price = bs_up_out_call_price(S, K, T, r, sigma, ub)
            print(
                f"  Error: {max([abs(analytical_call_option_price[i] - fair_price[i]) for i in range(len(fair_price))])}, "
                f"time elapsed: {end - start}")
