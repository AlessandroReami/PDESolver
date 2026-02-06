import math
from typing import List
from pyfreefem import FreeFemRunner


from PDESolver2.Common.DiscretizerType import DiscretizerType
from PDESolver2.Common.OptionType.OptionType import OptionType
from PDESolver2.Common.OptionType.PayoffType import PayoffType
from PDESolver2.Common.TimeMethodType import TimeMethodType
from PDESolver2.Discretizer.Discretizer import Discretizer
from PDESolver2.Dynamics.Dynamics import Dynamics
from PDESolver2.Option.Option import Option


class FEMDiscretizer(Discretizer):
    def __init__(self,  option: Option, dynamics: Dynamics, domain: List[List[float]], time_to_maturity: float,
                 time_steps: int, space_subdivision: List[int], time_method: TimeMethodType, abs_tol: float,
                 rel_tol: float, max_iter: int, lower_barriers: List[float], upper_barriers: List[float]):

        super().__init__(option, dynamics, domain, time_to_maturity, time_steps, space_subdivision, time_method, abs_tol, rel_tol,
                         max_iter)
        self._discretizer_type: DiscretizerType = DiscretizerType.FEM
        self._lower_barrier: List[float] = lower_barriers if lower_barriers is not None else [None for _ in
                                                                                              range(len(self._option.get_strike()))]
        self._upper_barrier: List[float] = upper_barriers if upper_barriers is not None else [None for _ in
                                                                                              range(len(self._option.get_strike()))]
        self._barrier_condition: str = ""
        condition_list = []
        if len(self._lower_barrier) == 1:
            variables = ["x"]
        elif len(self._lower_barrier) == 2:
            variables = ["x", "y"]
        elif len(self._lower_barrier) == 3:
            variables = ["x", "y", "z"]
        else:
            raise Exception("Not implemented")
        for lb, ub, variable in zip(self._lower_barrier, self._upper_barrier, variables):
            if lb != 0:
                condition_list.append(f"({variable} >= {lb})")
            if not math.isinf(ub):
                condition_list.append(f"({variable} <= {ub})")
        self._barrier_condition = "(" + "&&".join(condition_list) + ") ?"

    def get_discretizer_type(self):
        return self._discretizer_type

    def get_matrix(self, option_type: OptionType, dynamics: Dynamics):
        pass

    def get_payoff(self):
        raise Exception("Trova il modo di definire il payoff")

    def get_fair_price(self, recompute: bool = False):
        path = f"\\Test\\Test2Dim\\Plot\\{self._option.get_payoff_type().value}.png"
        if self._fair_price is not None and not recompute:
            return self._fair_price
        # if self._option.get_dimension().value == 1:
            # payoff = ""
            # if self._option.get_option_type().get_payoff_type() == PayoffType.PUT:
            #     payoff += f"max(K - x,0.)"
            #     boundary_condition = "on(0, u = K * exp(-r * (T - dt * int(y * M)))) + on(1, u = 0)"
            #
            # elif self._option.get_option_type().get_payoff_type() == PayoffType.CALL:
            #     payoff += f" max(x-K, 0.)"
            #     boundary_condition = "on(0, u = 0) + on(1, u = x * Smax - K * exp(-r * (T - dt * int(y * M))))"
            # else:
            #     raise Exception("Not developed")
            #
            # if self._barrier_condition != "() ?":
            #     payoff = self._barrier_condition + payoff + ": 0.;"
            # else:
            #     payoff += ";"
            #
            # code = f"""real sigma = $sigma; real r = $r; real K = $strike; real T = $T; int N = $space_steps;
            #                         int M = $time_steps; real Smax = $domain; real dt = $dt;
            #                         real dx = Smax / N;
            #
            #                         // Mesh
            #                         mesh Th = segment(N, [x * Smax, 0]);
            #
            #                         // Define finite element space
            #                         fespace Vh(Th, P1);
            #                         Vh uold, v;
            #                         Vh u = {payoff};
            #
            #                         // Define variational problem
            #                         solve BlackScholes(u, v) =
            #                             int1d(Th)((u - uold) * v / dt
            #                             + 0.5 * sigma^2 * x^2 * dx(u) * dx(v)
            #                             + r * x * dx(u) * v
            #                             - r * u * v)
            #                             + {boundary_condition};
            #
            #                         // Time stepping
            #                         for (int n = 1; n <= M; ++n) {{
            #                             uold = u; // Update previous solution
            #                             BlackScholes; // Solve the problem
            #                             plot(u, cmm = "Option Price at time step " + n);
            #                         }}
            #
            #                         // Update
            #                         j = j+1;
            #                     }};
            #
            #                     // Plot
            #                     plot(u, wait=true, value=true);"""
            #
            # FreeFemRunner(code, debug=0).execute({"strike": self._option.get_strike()[0],
            #                                       "domain": self._domain[0][1],
            #                                       "r": self._dynamics.get_interest_rate()[0][0],
            #                                       "dt": self._time_step_size,
            #                                       "sigma": self._dynamics.get_volatility()[0][0],
            #                                       "T": self._time_to_maturity,
            #                                       "time_steps": self._time_steps,
            #                                       "space_steps": self._space_subdivision[0]
            #                                       }, plot=True,
            #                                      verbosity=2)
        if self._option.get_dimension().value == 2:
            payoff = ""
            if self._option.get_payoff_type() == PayoffType.PUT_MIN:
                payoff += f"max(K-min(x,y),0.)"
                boundary_condition = "on(1, 4, u=K)"
            elif self._option.get_payoff_type() == PayoffType.PUT_MAX:
                payoff += f" max(K-max(x,y),0.)"
                boundary_condition = "on(2, 3, u=0)"
            elif self._option.get_payoff_type() == PayoffType.PUT_AVERAGE:
                payoff += f" max(K-(x+y), 0.)"
                boundary_condition = "on(2, 3, u=0)"

            elif self._option.get_payoff_type() == PayoffType.CALL_MIN:
                payoff += f" max(min(x,y)-K, 0.)"
                boundary_condition = "on(1, 4, u=0)"
            elif self._option.get_payoff_type() == PayoffType.CALL_MAX:
                payoff += f" max(max(x,y)-K,0.)"
                boundary_condition = "on(1, u=max(x-K, 0.)) + on(4, u=max(y-K, 0.))"
            elif self._option.get_payoff_type() == PayoffType.CALL_AVERAGE:
                payoff += f" max((x+y)-K, 0.)"
                boundary_condition = "on(1, u=max(x-K, 0.)) + on(4, u=max(y-K, 0.))"
            else:
                raise Exception("Not developed")
            if self._barrier_condition != "() ?":
                payoff = self._barrier_condition + payoff + ": 0.;"
            else:
                payoff += ";"

            # ma le condizioni ai bordi sono giuste??
            code = f"""// Parameters
                    int m1 = $m1; int m2 = $m2; int j = 100; real sigx = $sigx; real sigy = $sigy; real rho = $rho;
                    real r = $r; real dt = $dt; real K= $strike; real L= $domain_1; real LL = $domain_2;
                    real T = $T;

                    real rhohat=2*rho/(1+rho^2);

                    // Mesh
                    mesh th = square(m1, m2, [L*x, LL*y]);

                    // Fespace
                    fespace Vh(th, P1);
                    Vh u = {payoff}
                    Vh xveloc, yveloc, v, uold;

                    // Time loop
                    for (int n = 0; n*dt <= T; n++){{
                        // Mesh adaptation
                        if (j > 20){{
                            th = adaptmesh(th, u, verbosity=1, abserror=1, nbjacoby=2,
                            err=0.001, nbvx=5000, omega=1.8, ratio=1.8, nbsmooth=3,
                            splitpbedge=1, maxsubdiv=5, rescaling=1);
                            j = 0;
                            xveloc = -x*r + x*sigx^2 + x*rhohat*sigx*sigy/2;
                            yveloc = -y*r + y*sigy^2 + y*rhohat*sigx*sigy/2;
                            u = u;
                        }}

                        // Update
                        uold = u;

                        // Solve
                        solve eq1(u, v, init=j, solver=LU)
                            = int2d(th)(
                                  u*v*(r+1/dt)
                                + dx(u)*dx(v)*(x*sigx)^2/2
                                + dy(u)*dy(v)*(y*sigy)^2/2
                                + (dy(u)*dx(v) + dx(u)*dy(v))*rhohat*sigx*sigy*x*y/2
                            )
                            - int2d(th)(
                                  v*convect([xveloc, yveloc], dt, uold)/dt
                            )
                            + {boundary_condition}
                            ;

                        // Update
                        j = j+1;
                    }};

                    // Plot
                    plot(u, wait=true, fill=1, value=true, aspectratio= true);"""

            FreeFemRunner(code, debug=0).execute({"strike": self._option.get_strike()[0],
                                                  "domain_1": self._domain[0][1],
                                                  "domain_2": self._domain[1][1],
                                                  "r": self._dynamics.get_interest_rate()[0][0][0],
                                                  "dt": self._time_step_size,
                                                  "sigx": self._dynamics.get_volatility()[0][0][0],
                                                  "sigy": self._dynamics.get_volatility()[0][0][0],
                                                  "rho": self._dynamics.get_correlation(),
                                                  "m1": self._space_subdivision[0],
                                                  "m2": self._space_subdivision[1],
                                                  "T": self._time_to_maturity
                                                  }, plot=True,
                                                 verbosity=-1)
        else:
            raise Exception("Not developed")
        return self._fair_price

    def plot_fair_price(self, path: str, color: str = "blue", label: str = "Option price", plot_payoff: bool = False,
                        plot_analytical: bool= True):
        pass
        # raise Exception(f"Plot not developed for FEM option")
