import time
from typing import List

from numpy import ndarray

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
from PDESolver2.UsefulFunction.UsefulFunction import black_scholes_price, plot_error_decay, plot_pointwise_error, \
    plot_embedded_error


def test_space_decay(space_subdivision: List[List[int]], time_steps: List[int],
                     reference_order_of_convergence: int, vol: float, r: float, dividend: float,
                     discretizer_type: FiniteDifferencesType, time_method_type: TimeMethodType):
    print(f"Testing space error decay {discretizer_type.value}, {time_method_type.value}")

    # Constant terms
    builder = Builder()
    volatility = ConstantVolatility(vol)
    interest_rate = ConstantInterestRate(r)
    dividend_rate = ConstantDividendRate(dividend)
    dynamics_type = DynamicsType.BLACK_SCHOLES
    dynamics = builder.build_dynamics(dynamics_type, volatility, interest_rate, dividend_rate)

    underlying_values = [100]
    strike = [100]
    option_type = OptionType(DimensionType.DIM1, ExerciseType.EUROPEAN, PayoffType.CALL)
    _ = builder.build_option(underlying_values, strike, option_type)

    domain = [[0.1, 200]]
    time_to_maturity = 1
    errors = []

    for ts, ss in zip(time_steps, space_subdivision):
        discretizer = builder.build_discretizer(discretizer_type, domain, time_to_maturity, ts, ss,
                                                time_method_type, abs_tol=1e-5, rel_tol=1e-5, max_iter=100)
        start = time.time()
        fair_price = discretizer.get_fair_price()
        end = time.time()
        # variables useful to compute the analytical price
        K = strike[0]
        T = time_to_maturity
        r = dynamics.get_interest_rate()[0][0]
        q = dynamics.get_dividend_rate()[0][0]
        sigma = dynamics.get_volatility()[0][0]
        S = discretizer.get_discretized_domain()[0]

        analytical_call_option_price = black_scholes_price(option_type.get_payoff_type(), S, K, T, r, sigma, q)

        path = f'Test\\TestCall1D\\{time_method_type.value}\\Space\\PW_error\\t{ts}_s{ss[0]}_order_{discretizer_type.value}.png'
        plot_pointwise_error(S, fair_price, analytical_call_option_price,
                             "Point wise error between analytical and estimated solution", path,
                             "Point wise error", "Underlying value")
        errors.append(max([abs(analytical_call_option_price[i] - fair_price[i]) for i in range(len(fair_price))]))
        path = f'Test\\TestCall1D\\{time_method_type.value}\\Space\\Compare_Analytical_sol\\t{ts}_s{ss[0]}_order_{discretizer_type.value}.png'
        discretizer.plot_fair_price(path)
        string = f"Error: {errors[-1]}, elapsed time: {end - start}"
        print(string)

    title = (f"Error decay for the {time_method_type.value} on a {option_type.get_exercise_type().value}"
             f" {option_type.get_payoff_type().value}")
    label = "Error decay"
    x_label = "Log space subdivision"

    space_subdivision = [sub[0] for sub in space_subdivision]
    path = f"Test\\TestCall1D\\{time_method_type.value}\\Space\\space_decay_order_{discretizer_type.value}.png"
    plot_error_decay(space_subdivision, errors, reference_order_of_convergence, title, label, x_label, path)


def test_space_decay_reference_sol(space_subdivision: List[List[int]], time_steps: List[int],
                                   reference_order_of_convergence: int, reference_space_subdivision: List[int],
                                   reference_time_steps: int,
                                   vol: float, r: float, dividend: float,
                                   discretizer_type: FiniteDifferencesType,
                                   time_method_type: TimeMethodType):
    print(f"Testing space error decay {discretizer_type.value}, {time_method_type.value} with reference sol")

    # Constant terms
    builder = Builder()
    volatility = ConstantVolatility(vol)
    interest_rate = ConstantInterestRate(r)
    dividend_rate = ConstantDividendRate(dividend)
    dynamics_type = DynamicsType.BLACK_SCHOLES
    _ = builder.build_dynamics(dynamics_type, volatility, interest_rate, dividend_rate)

    underlying_values = [100]
    strike = [100]
    option_type = OptionType(DimensionType.DIM1, ExerciseType.EUROPEAN, PayoffType.CALL)
    _ = builder.build_option(underlying_values, strike, option_type)

    domain = [[0.1, 200]]
    time_to_maturity = 1
    errors = []

    # compute reference sol
    ref_discretizer = builder.build_discretizer(discretizer_type, domain, time_to_maturity, reference_time_steps,
                                                reference_space_subdivision, time_method_type, abs_tol=1e-5,
                                                rel_tol=1e-5, max_iter=100)
    reference_price = ref_discretizer.get_fair_price()

    for ts, ss in zip(time_steps, space_subdivision):
        discretizer = builder.build_discretizer(discretizer_type, domain, time_to_maturity, ts, ss,
                                                time_method_type, abs_tol=1e-5, rel_tol=1e-5, max_iter=100)
        start = time.time()
        fair_price = discretizer.get_fair_price()
        end = time.time()
        S = discretizer.get_discretized_domain()[0]

        interpolated_ref_price = interpolate(S, reference_price, ref_discretizer.get_discretized_domain()[0])

        path = (f'Test\\TestCall1D\\{time_method_type.value}\\Space\\PW_Error\\t{ts}_s{ss[0]}_reference'
                f'_t{reference_time_steps}_s{reference_space_subdivision[0]}_order_{discretizer_type.value}.png')
        plot_pointwise_error(S, fair_price, interpolated_ref_price,
                             "Point wise error between reference and estimated solution", path,
                             "Point wise error", "Underlying value")
        errors.append(max([abs(interpolated_ref_price[i] - fair_price[i]) for i in range(len(fair_price))]))
        path = (f'Test\\TestCall1D\\{time_method_type.value}\\Space\\Compare_Reference_sol\\t{ts}_s{ss[0]}_reference'
                f'_t{reference_time_steps}_s{reference_space_subdivision[0]}_order_{discretizer_type.value}.png')
        discretizer.plot_fair_price(path)
        print(f"Error: {errors[-1]}, elapsed time: {end - start}")

    title = (f"Error decay for the {time_method_type.value} on a {option_type.get_exercise_type().value}"
             f" {option_type.get_payoff_type().value}")
    label = "Error decay"
    x_label = "Log space subdivision"

    space_subdivision = [sub[0] for sub in space_subdivision]
    path = (f"Test\\TestCall1D\\{time_method_type.value}\\Space\\space_decay_reference"
            f"_t{reference_time_steps}_s{reference_space_subdivision[0]}_order_{discretizer_type.value}.png")
    plot_error_decay(space_subdivision, errors, reference_order_of_convergence, title, label, x_label, path)


def test_time_decay_reference_sol(space_subdivision: List[List[int]], time_steps: List[int],
                                  reference_order_of_convergence: int, reference_space_subdivision: List[int],
                                  reference_time_steps: int,
                                  vol: float, r: float, dividend: float,
                                  discretizer_type: FiniteDifferencesType,
                                  time_method_type: TimeMethodType):
    print(f"Testing space error decay {discretizer_type.value}, {time_method_type.value} with reference sol")

    # Constant terms
    builder = Builder()
    volatility = ConstantVolatility(vol)
    interest_rate = ConstantInterestRate(r)
    dividend_rate = ConstantDividendRate(dividend)
    dynamics_type = DynamicsType.BLACK_SCHOLES
    _ = builder.build_dynamics(dynamics_type, volatility, interest_rate, dividend_rate)

    underlying_values = [100]
    strike = [100]
    option_type = OptionType(DimensionType.DIM1, ExerciseType.EUROPEAN, PayoffType.CALL)
    _ = builder.build_option(underlying_values, strike, option_type)

    domain = [[0.1, 200]]
    time_to_maturity = 1
    errors = []

    # compute reference sol
    ref_discretizer = builder.build_discretizer(discretizer_type, domain, time_to_maturity, reference_time_steps,
                                                reference_space_subdivision, time_method_type, abs_tol=1e-7,
                                                rel_tol=1e-7, max_iter=100)
    reference_price = ref_discretizer.get_fair_price()

    for ts, ss in zip(time_steps, space_subdivision):
        discretizer = builder.build_discretizer(discretizer_type, domain, time_to_maturity, ts, ss,
                                                time_method_type, abs_tol=1e-5, rel_tol=1e-5, max_iter=100)
        start = time.time()
        fair_price = discretizer.get_fair_price()
        end = time.time()
        S = discretizer.get_discretized_domain()[0]

        interpolated_ref_price = interpolate(S, reference_price, ref_discretizer.get_discretized_domain()[0])

        path = (f'Test\\TestCall1D\\{time_method_type.value}\\Time\\PW_error\\t{ts}_s{ss[0]}_reference'
                f' t{reference_time_steps} s{reference_space_subdivision[0]} order {discretizer_type.value}.png')
        plot_pointwise_error(S, fair_price, interpolated_ref_price,
                             "Point wise error between reference and estimated solution", path,
                             "Point wise error", "Underlying value")
        errors.append(max([abs(interpolated_ref_price[i] - fair_price[i]) for i in range(len(fair_price))]))
        path = (f'Test\\TestCall1D\\{time_method_type.value}\\Time\\Compare_Reference_sol\\t{ts}_s{ss[0]}_reference'
                f'_t{reference_time_steps}_s{reference_space_subdivision[0]}_order_{discretizer_type.value}.png')
        discretizer.plot_fair_price(path)
        print(f"Error: {errors[-1]}, elapsed time: {end - start}")

    title = (f"Error decay for the {time_method_type.value} on a {option_type.get_exercise_type().value}"
             f" {option_type.get_payoff_type().value}")
    label = "Error decay"
    x_label = "Log-timesteps"

    path = (f"Test\\TestCall1D\\{time_method_type.value}\\Time\\Time_decay_reference"
            f"_t{reference_time_steps}_s{reference_space_subdivision[0]}_order_{discretizer_type.value}.png")
    plot_error_decay(time_steps, errors, reference_order_of_convergence, title, label, x_label, path)


def interpolate(domain: ndarray, reference_sol: ndarray, reference_domain: ndarray):
    index = 0
    interpolated_sol = []
    for x in domain:
        while reference_domain[index + 1] < x:
            index += 1
        x_0, x_1 = reference_domain[index: index + 2]
        y_0, y_1 = reference_sol[index: index + 2]
        interpolated_point = (x - x_0) / (x_1 - x_0) * (y_1 - y_0) + y_0
        interpolated_sol.append(interpolated_point)
    return interpolated_sol


def test_time_decay(space_subdivision: List[List[int]], time_steps: List[int],
                    reference_order_of_convergence: int, vol: float, r: float, dividend: float,
                    discretizer_type: FiniteDifferencesType, time_method_type: TimeMethodType):
    print(f"Testing time error decay {discretizer_type.value}, {time_method_type.value}")

    # Constant terms
    builder = Builder()
    volatility = ConstantVolatility(vol)
    interest_rate = ConstantInterestRate(r)
    dividend_rate = ConstantDividendRate(dividend)
    dynamics_type = DynamicsType.BLACK_SCHOLES
    dynamics = builder.build_dynamics(dynamics_type, volatility, interest_rate, dividend_rate)

    underlying_values = [100]
    strike = [100]
    option_type = OptionType(DimensionType.DIM1, ExerciseType.EUROPEAN, PayoffType.CALL)
    _ = builder.build_option(underlying_values, strike, option_type)

    domain = [[0.1, 200]]
    time_to_maturity = 1

    errors = []

    for ts, ss in zip(time_steps, space_subdivision):
        discretizer = builder.build_discretizer(discretizer_type, domain, time_to_maturity, ts, ss,
                                                time_method_type, abs_tol=1e-7, rel_tol=1e-7, max_iter=100)
        start = time.time()
        fair_price = discretizer.get_fair_price()
        end = time.time()
        # variables useful to compute the analytical price
        K = strike[0]
        T = time_to_maturity
        r = dynamics.get_interest_rate()[0][0]
        q = dynamics.get_dividend_rate()[0][0]
        sigma = dynamics.get_volatility()[0][0]
        S = discretizer.get_discretized_domain()[0]

        analytical_call_option_price = black_scholes_price(option_type.get_payoff_type(), S, K, T, r, sigma, q)

        path = f'Test\\TestCall1D\\{time_method_type.value}\\Time\\PW_error\\t{ts}_s{ss[0]}_order_{discretizer_type.value}.png'
        plot_pointwise_error(S, fair_price, analytical_call_option_price,
                             "Point wise error between analytical and estimated solution", path,
                             "Point wise error", "Underlying value")
        errors.append(max([abs(analytical_call_option_price[i] - fair_price[i]) for i in range(len(fair_price))]))
        path = f'Test\\TestCall1D\\{time_method_type.value}\\Time\\Compare_Analytical_sol\\t{ts}_s{ss[0]}_order_{discretizer_type.value}.png'
        discretizer.plot_fair_price(path)
        print(f"Error: {errors[-1]}, elapsed time: {end - start}")

    title = (f"Error decay for the {time_method_type.value} on a {option_type.get_exercise_type().value}"
             f" {option_type.get_payoff_type().value}")
    label = "Error decay"
    x_label = "Log-timesteps"

    path = f"Test\\TestCall1D\\{time_method_type.value}\\Time\\Time_decay_order_{discretizer_type.value}.png"
    plot_error_decay(time_steps, errors, reference_order_of_convergence, title, label, x_label, path)


def test_embedded_method(space_subdivision: List[int], time_steps: int, tol_list: List[List[float]],
                         reference_tol: List[float], vol: float, r: float, dividend: float,
                         discretizer_type: FiniteDifferencesType, time_method_type: TimeMethodType):
    print(f"Testing embedding method: {time_method_type.value}")

    # Constant terms
    builder = Builder()
    volatility = ConstantVolatility(vol)
    interest_rate = ConstantInterestRate(r)
    dividend_rate = ConstantDividendRate(dividend)
    dynamics_type = DynamicsType.BLACK_SCHOLES
    dynamics = builder.build_dynamics(dynamics_type, volatility, interest_rate, dividend_rate)

    underlying_values = [100]
    strike = [100]
    option_type = OptionType(DimensionType.DIM1, ExerciseType.EUROPEAN, PayoffType.CALL)
    _ = builder.build_option(underlying_values, strike, option_type)

    domain = [[0.1, 200]]
    time_to_maturity = 1
    errors = []
    discretizer = builder.build_discretizer(discretizer_type, domain, time_to_maturity, time_steps, space_subdivision,
                                            time_method_type, abs_tol=reference_tol[0], rel_tol=reference_tol[1], max_iter=1000)
    reference_sol = discretizer.get_fair_price()

    for at, rt in tol_list:
        discretizer = builder.build_discretizer(discretizer_type, domain, time_to_maturity, time_steps, space_subdivision,
                                                time_method_type, abs_tol=at, rel_tol=rt, max_iter=1000)
        start = time.time()
        fair_price = discretizer.get_fair_price()
        end = time.time()
        errors.append(max([abs(reference_sol[i] - fair_price[i]) for i in range(len(fair_price))]))
        string = f"abs_tol= {at}, rel_tol= {rt} Error: {errors[-1]}, elapsed time: {end - start}"
        print(string)

    title = f"Error with different tolerances"
    label = "Error over the different run"
    x_label = "Tolerance"

    path = f"Test\\TestCall1D\\{time_method_type.value}\\Space\\space decay order {discretizer_type.value}.png"
    abs_tol = [at for at, rt in tol_list]
    plot_embedded_error(abs_tol, errors, title, label, x_label, path)
