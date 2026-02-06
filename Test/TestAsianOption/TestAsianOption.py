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

test_array = [PayoffType.ASIAN_CALL]


def test_simple_asian_option():
    for p in test_array:
        start = time.time()
        builder = Builder()
        print(f"{p.value} option")
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
        _ = builder.build_option(underlying_values, strike, option_type)

        discretizer_type = FiniteDifferencesType.ORDER2
        domain = [[-1, 1]]
        time_to_maturity = 1
        time_steps = 1000
        space_subdivision = [500]
        time_method_type = TimeMethodType.CRANK_NICOLSON

        discretizer = builder.build_discretizer(discretizer_type, domain, time_to_maturity, time_steps,
                                                space_subdivision,
                                                time_method_type, abs_tol=1e-5, rel_tol=1e-5, max_iter=100)
        fair_price = discretizer.get_fair_price()
        end = time.time()
        print(f"Elapsed time: {end-start}")
        path = f"Test\\TestAsianOption\\{option_type.get_payoff_type().value}\\{option_type.get_payoff_type().value}.png"
        discretizer.plot_fair_price(path=path, color="blue", plot_payoff=False, plot_analytical=False)
