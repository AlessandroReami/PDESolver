from PDESolver2.Common.FiniteDifferencesType import FiniteDifferencesType
from PDESolver2.Common.TimeMethodType import TimeMethodType
from PDESolver2.Test.Test2Dim.Test2Dim import test_2dim_option
from PDESolver2.Test.TestAsianOption.TestAsianOption import test_simple_asian_option
from PDESolver2.Test.TestBarrierOption.TestBarrierOption import test_simple_barrier_option
from PDESolver2.Test.TestCall1D.Test_call_1_d_space_decay import test_space_decay, test_space_decay_reference_sol, \
    test_time_decay, test_time_decay_reference_sol, test_embedded_method
from PDESolver2.Test.TestSimpleOption.TestSimpleOption import test_simple_option
from PDESolver2.Test.TestTimeMethod.TestTimeMethod import test_time_method

###
# test_simple_option()
# test_time_method()
# test_simple_barrier_option(0)
# test_simple_barrier_option(1)
# test_simple_asian_option()
# test_2dim_option()

vol = 0.2
r = 0.03
dividend = 0

# ### Backward Euler test
# discretizer_type = FiniteDifferencesType.ORDER4
# time_method_type = TimeMethodType.BACKWARD_EULER
#
# space_subdivision = [[50], [100], [150], [200], [250], [300]]
# time_steps = [10000] * len(space_subdivision)
# test_space_decay(space_subdivision, time_steps, 2, vol, r, dividend,
#                  discretizer_type, time_method_type)
# test_space_decay_reference_sol(space_subdivision, time_steps, 2,
#                                [1000], 10000, vol, r, dividend,
#                                discretizer_type, time_method_type)
#
# time_steps = [50, 100, 200, 400]
# space_subdivision = [[1000]] * len(time_steps)
# test_time_decay(space_subdivision, time_steps, 1, vol, r, dividend, discretizer_type,
#                 time_method_type)
# test_time_decay_reference_sol(space_subdivision, time_steps, 1,
#                               [1000], 10000, vol, r, dividend,
#                               discretizer_type, time_method_type)


# ## Krank Nicolson test
discretizer_type = FiniteDifferencesType.ORDER4
time_method_type = TimeMethodType.CRANK_NICOLSON
# space_subdivision = [[50], [100], [150], [200], [250], [300]]
# time_steps = [10000] * len(space_subdivision)
# test_space_decay(space_subdivision, time_steps, 2, vol, r, dividend,
#                  discretizer_type,
#                  time_method_type)
# test_space_decay_reference_sol(space_subdivision, time_steps, 2,
#                                [1000], 10000, vol, r, dividend,
#                                discretizer_type, time_method_type)
time_steps = [50, 100, 175, 250]
space_subdivision = [[4001]] * len(time_steps)
test_time_decay(space_subdivision, time_steps, 2, vol, r, dividend, discretizer_type,
                time_method_type)
test_time_decay_reference_sol(space_subdivision, time_steps, 2,
                              [4001], 10000, vol, r, dividend,
                              discretizer_type, time_method_type)



### Diagonally Implicit Runge Kutta 3 test
discretizer_type = FiniteDifferencesType.ORDER2
time_method_type = TimeMethodType.RUNGE_KUTTA_3_SEMI_IMPL

space_subdivision = [[50], [100], [150], [200], [250], [300]]
time_steps = [10000] * len(space_subdivision)
test_space_decay(space_subdivision, time_steps, 2, vol, r, dividend,
                 discretizer_type,
                 time_method_type)
test_space_decay_reference_sol(space_subdivision, time_steps, 2,
                               [1000], 10000, vol, r, dividend,
                               discretizer_type, time_method_type)
time_steps = [50, 75, 100]
space_subdivision = [[8001]] * len(time_steps)
test_time_decay(space_subdivision, time_steps, 3, vol, r, dividend, discretizer_type,
                time_method_type)
test_time_decay_reference_sol(space_subdivision, time_steps, 3,
                              [8001], 1001, vol, r, dividend,
                              discretizer_type, time_method_type)


# ## Embedded Runge Kutta Lobatto 3C test
discretizer_type = FiniteDifferencesType.ORDER2
time_method_type = TimeMethodType.RUNGE_KUTTA_LOBATTO3C

space_subdivision = [[50], [100], [150], [200], [250], [300]]
time_steps = [10000] * len(space_subdivision)
test_space_decay(space_subdivision, time_steps, 2, vol, r, dividend,
                 discretizer_type,
                 time_method_type)
test_space_decay_reference_sol(space_subdivision, time_steps, 2,
                               [1000], 10000, vol, r, dividend,
                               discretizer_type, time_method_type)

time_steps = [175, 200, 225, 250]
space_subdivision = [[501]] * len(time_steps)
test_time_decay(space_subdivision, time_steps, 3, vol, r, dividend, discretizer_type,
                time_method_type)
test_time_decay_reference_sol(space_subdivision, time_steps, 3,
                              [501], 501, vol, r, dividend,
                              discretizer_type, time_method_type)



## Embedded Runge Kutta Lobatto 3B test
discretizer_type = FiniteDifferencesType.ORDER2
time_method_type = TimeMethodType.RUNGE_KUTTA_LOBATTO3B

space_subdivision = [[50], [100], [150], [200], [250], [300]]
time_steps = [10000] * len(space_subdivision)
test_space_decay(space_subdivision, time_steps, 2, vol, r, dividend,
                 discretizer_type,
                 time_method_type)
test_space_decay_reference_sol(space_subdivision, time_steps, 2,
                               [1000], 10000, vol, r, dividend,
                               discretizer_type, time_method_type)

time_steps = [150, 175, 200, 225, 275, 325]
space_subdivision = [[401]] * len(time_steps)
test_time_decay(space_subdivision, time_steps, 3, vol, r, dividend, discretizer_type,
                time_method_type)
test_time_decay_reference_sol(space_subdivision, time_steps, 3,
                              [401], 1001, vol, r, dividend,
                              discretizer_type, time_method_type)


# Explict Embedded Runge Kutta 2 test
discretizer_type = FiniteDifferencesType.ORDER2
time_method_type = TimeMethodType.RUNGE_KUTTA_2_EMBEDDED

space_subdivision = [500]
time_steps = 10000
tol_list = [[1e-2, 1e-4], [1e-3, 1e-5], [1e-4, 1e-6], [1e-5, 1e-7], [1e-6, 1e-8], [1e-7, 1e-9]]
reference_tol= [1e-10, 1e-10]
test_embedded_method(space_subdivision, time_steps, tol_list, reference_tol, vol, r, dividend,
                     discretizer_type, time_method_type)
