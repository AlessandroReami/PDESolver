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

builder = Builder()

volatility = ConstantVolatility(0.2)
interest_rate = ConstantInterestRate(0.03)
dividend_rate = ConstantDividendRate(0.0)
correlation = 0.3
dynamics_type = DynamicsType.BLACK_SCHOLES
dynamics = builder.build_dynamics(dynamics_type, volatility, interest_rate, dividend_rate, correlation=correlation)

underlying_values = [100]
strike = [1]
option_type = OptionType(DimensionType.DIM1, ExerciseType.EUROPEAN, PayoffType.ASIAN_CALL)
option = builder.build_option(underlying_values, strike, option_type)

discretizer_type = FiniteDifferencesType.ORDER2
domain = [[-2, 2]]
time_to_maturity = 1
time_steps = 100
space_subdivision = [300]
time_method_type = TimeMethodType.BACKWARD_EULER

discretizer = builder.build_discretizer(discretizer_type, domain, time_to_maturity, time_steps, space_subdivision,
                                        time_method_type, abs_tol=1e-5, rel_tol=1e-5, max_iter=100)

# for fem:
# option = builder.build_option(dynamics, underlying_values, strike, option_type, upper_barriers=[100, 100], lower_barriers=[0,0])

#lower_barriers = [0]
#upper_barriers = [np.inf]
#option = builder.build_option(dynamics, underlying_values, strike, option_type, lower_barriers, upper_barriers)
# option.compute_average_time_needed(40)
fair_price = discretizer.get_fair_price()
discretizer.plot_fair_price(path=f"sigma {dynamics.get_volatility()[0][0]} r {dynamics.get_interest_rate()[0][0]}"
                                 f" q {dynamics.get_dividend_rate()[0][0]}.png", color="blue", plot_payoff=False,
                            plot_analytical=False)

# S = discretizer.get_discretized_domain()[0]
# K = strike[0]
# T = time_to_maturity
# r = dynamics.get_interest_rate()[0][0]
# q = dynamics.get_dividend_rate()[0][0]
# sigma = dynamics.get_volatility()[0][0]
#
# analytical_call_option_price = black_scholes_price(option_type.get_payoff_type(), S, K, T, r, sigma, q)

#print(f"Error: {max([abs(analytical_call_option_price[i] - fair_price[i]) for i in range(len(fair_price))])}")

# 3B
# Execution time: 0.04899907112121582
# Error: [0.00259242]

# Process finished with exit code 0

# 3C
# Error: [9.98240777e-05]
