import time

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
from PDESolver2.UsefulFunction.UsefulFunction import black_scholes_price

test_array = [TimeMethodType.FORWARD_EULER,
              TimeMethodType.BACKWARD_EULER,
              TimeMethodType.CRANK_NICOLSON,
              TimeMethodType.RUNGE_KUTTA_2_EMBEDDED,
              TimeMethodType.RUNGE_KUTTA_3_SEMI_IMPL,
              TimeMethodType.RUNGE_KUTTA_LOBATTO3C,
              TimeMethodType.RUNGE_KUTTA_LOBATTO3B
              ]


def test_time_method():
    for t in test_array:
        start = time.time()
        builder = Builder()
        print(f"Time method: {t.value} ")
        volatility = ConstantVolatility(0.2)
        interest_rate = ConstantInterestRate(0.03)
        dividend_rate = ConstantDividendRate(0)
        correlation = 0.3
        dynamics_type = DynamicsType.BLACK_SCHOLES
        dynamics = builder.build_dynamics(dynamics_type, volatility, interest_rate, dividend_rate,
                                          correlation=correlation)

        underlying_values = [100]
        strike = [100]
        option_type = OptionType(DimensionType.DIM1, ExerciseType.EUROPEAN, PayoffType.CALL)
        option = builder.build_option(underlying_values, strike, option_type)

        discretizer_type = FiniteDifferencesType.ORDER4
        domain = [[0.1, 200]]
        time_to_maturity = 1
        time_steps = 2000
        space_subdivision = [300]
        time_method_type = t

        discretizer = builder.build_discretizer(discretizer_type, domain, time_to_maturity, time_steps,
                                                space_subdivision,
                                                time_method_type, abs_tol=1e-5, rel_tol=1e-5, max_iter=100)
        fair_price = discretizer.get_fair_price()
        end = time.time()
        path = (f"Test\\TestTimeMethod\\Results\\{t.value}.png")
        discretizer.plot_fair_price(path=path, color="blue", plot_payoff=True)

        S = discretizer.get_discretized_domain()[0]
        K = strike[0]
        T = time_to_maturity
        r = dynamics.get_interest_rate()[0][0]
        q = dynamics.get_dividend_rate()[0][0]
        sigma = dynamics.get_volatility()[0][0]

        analytical_call_option_price = black_scholes_price(option_type.get_payoff_type(), S, K, T, r, sigma, q)

        print(f"  Error: {max([abs(analytical_call_option_price[i] - fair_price[i]) for i in range(len(fair_price))])}, "
              f"time elapsed: {end - start}")
