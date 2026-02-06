from array import array
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix

import math

from PDESolver2.Common.OptionType.PayoffType import PayoffType


def element_by_row_mult(vec: array, matrix_test: csr_matrix):
    result = matrix_test.copy()
    result.data *= vec.repeat(np.diff(result.indptr))
    return result


def black_scholes_price(payoff: PayoffType, S: np.ndarray, K: float, T: float, r: float, sigma: float, q: float,
                        l: float = 0, u: float = np.infty):
    price = np.zeros((len(S)))
    for i in range(len(S)):
        d1 = (math.log(S[i] / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if payoff == PayoffType.CALL:
            price[i] = S[i] * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
        elif payoff == PayoffType.PUT:
            price[i] = K * math.exp(-r * T) * norm_cdf(-d2) - S[i] * math.exp(-q * T) * norm_cdf(-d1)
        elif payoff == PayoffType.DIGITAL_CALL:
            price[i] = math.exp(-r * T) * norm_cdf(d2)
        elif payoff == PayoffType.DIGITAL_PUT:
            price[i] = math.exp(-r * T) * norm_cdf(-d2)
    return price


def bs_down_out_put_price(S: np.ndarray, K: float, T: float, r: float, sigma: float, b: float):
    price = np.zeros((len(S)))
    for i in range(len(S)):
        s = S[i]
        price[i] = K * BOb(T, r, sigma, s, b) - STb(T, r, sigma, s, b) + C_down(T, r, sigma, s, b, K)
    return price


def bs_up_out_call_price(S: np.ndarray, K: float, T: float, r: float, sigma: float, b: float):
    price = np.zeros((len(S)))
    for i in range(len(S)):
        s = S[i]
        price[i] = - K * BOB(T, r, sigma, s, b) + STB(T, r, sigma, s, b) + P_up(T, r, sigma, s, b, K)
    return price


def bs_up_out_call_price_chat(S: np.ndarray, K: float, T: float, r: float, sigma: float, b: float):
    price = np.zeros((len(S)))
    for i in range(len(S)):
        s = S[i]
        d1 = (math.log(s / b) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        price[i] = C(T, r, sigma, s, K) - s * (b / s) ** (2 * (r / sigma ** 2 - 0.5)) * norm_cdf(d1) + K * math.exp(
            -r * T) * (b / s) ** (2 * (r / sigma ** 2 - 0.5) - 1) * norm_cdf(d2)
    return price


def H(T, r, sigma, s, b):
    # digital call price
    r_tilde = r - 1 / 2 * sigma ** 2
    return math.exp(-r * (T)) * norm_cdf((r_tilde * (T) + math.log(s / b)) / (sigma * math.sqrt(T)))


def BOb(T, r, sigma, s, b):
    r_tilde = r - 1 / 2 * sigma ** 2
    return H(T, r, sigma, s, b) - (b / s) ** (2 * r_tilde / sigma ** 2) * H(T, r, sigma, b ** 2 / s, b)


def C(T, r, sigma, s, K):
    d1 = (math.log(s / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return s * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def P(T, r, sigma, s, K):
    d1 = (math.log(s / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return -s * norm_cdf(-d1) + K * math.exp(-r * T) * norm_cdf(-d2)


def STb(T, r, sigma, s, b):
    r_tilde = r - 1 / 2 * sigma ** 2
    return (b * (H(T, r, sigma, s, b) - (b / s) ** (2 * r_tilde / sigma ** 2) * H(T, r, sigma, b ** 2 / s, b))
            + C(T, r, sigma, s, b) - (b / s) ** (2 * r_tilde / sigma ** 2) * C(T, r, sigma, b ** 2 / s, b))


def HB(T, r, sigma, s, b):
    # digital put price
    r_tilde = r - 1 / 2 * sigma ** 2
    return math.exp(-r * (T)) * norm_cdf(-(r_tilde * (T) + math.log(s / b)) / (sigma * math.sqrt(T)))


def BOB(T, r, sigma, s, b):
    r_tilde = r - 1 / 2 * sigma ** 2
    return HB(T, r, sigma, s, b) - (b / s) ** (2 * r_tilde / sigma ** 2) * HB(T, r, sigma, b ** 2 / s, b)


def STB(T, r, sigma, s, b):
    r_tilde = r - 1 / 2 * sigma ** 2
    return (b * (HB(T, r, sigma, s, b) - (b / s) ** (2 * r_tilde / sigma ** 2) * HB(T, r, sigma, b ** 2 / s, b))
            - P(T, r, sigma, s, b) + (b / s) ** (2 * r_tilde / sigma ** 2) * P(T, r, sigma, b ** 2 / s, b))


def P_up(T, r, sigma, s, b, K):
    r_tilde = r - 1 / 2 * sigma ** 2
    if b > K:
        return P(T, r, sigma, s, K) - (b / s) ** (2 * r_tilde / sigma ** 2) * P(T, r, sigma, b ** 2 / s, K)
    else:
        raise Exception("Not implemented")


def C_down(T, r, sigma, s, b, K):
    r_tilde = r - 1 / 2 * sigma ** 2
    if b < K:
        return C(T, r, sigma, s, K) - (b / s) ** (2 * r_tilde / sigma ** 2) * C(T, r, sigma, b ** 2 / s, K)
    else:
        raise Exception("Not implemented")


def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def black_scholes_up_and_out(payoff: PayoffType, S: np.ndarray, K: float, T: float, r: float, sigma: float, q: float,
                             B: float):
    price = np.zeros(len(S))
    for i in range(len(S)):
        if payoff == PayoffType.CALL:
            #todo errata
            raise Exception("manca")
        elif payoff == PayoffType.PUT:
            #è davvero la cdf?
            H = math.exp(-r * T) * norm_cdf(((r - 0.5 * sigma ** 2) * T + math.log(S[i])) / (sigma * math.sqrt(T)))
            H_2 = math.exp(-r * T) * norm_cdf(
                ((r - 0.5 * sigma ** 2) * T + math.log(B ** 2 / S[i])) / (sigma * math.sqrt(T)))
            rr = (2 * r - sigma ** 2) / sigma ** 2
            price[i] = (black_scholes_price(PayoffType.PUT, S, K, T, r, sigma, q) - (K - B) * H - (B / S[i]) ** rr *
                        (black_scholes_price(PayoffType.PUT, B ** 2 / S[i], K, T, r, sigma, q) - (K - B) * H_2) +
                        (1 - (B / S[i]) ** rr) * (K - B) * math.exp(-r * T))
    return price


def black_scholes_down_and_out(payoff: PayoffType, S: np.ndarray, K: float, T: float, r: float, sigma: float, q: float,
                               B: float):
    price = np.zeros(len(S))
    B_s = np.array([B / s for s in S])
    B2_s = np.array([(B ** 2) / s for s in S])
    if payoff == PayoffType.CALL:
        #controlla
        price = black_scholes_price(PayoffType.CALL, S, K, T, r, sigma, q) - B_s ** (
                (2 * r - sigma ** 2) / sigma ** 2) * black_scholes_price(PayoffType.CALL, B2_s, K, T, r, sigma, q)
    elif payoff == PayoffType.PUT:
        #todo errata
        raise Exception("Non sviluppata")
    return price


def plot_error_decay(x_values: List[float], errors: List[float], reference_order_of_convergence: int, title: str,
                     label: str, x_label: str, path: str, color: str = "blue"):
    title_fontsize = 17
    label_fontsize = 11
    figsize = (8, 6)
    color_refenrence = "red"
    label_reference = f"Order of convergence {reference_order_of_convergence}"

    plt.figure(figsize=figsize)
    plt.loglog(x_values, errors, color=color, label=label)
    array = [errors[-1] * (val / x_values[-1]) ** (-reference_order_of_convergence) for val in x_values]
    plt.loglog(x_values, array,
               color=color_refenrence, label=label_reference)
    plt.legend()
    plt.xlabel(x_label, fontsize=label_fontsize)
    plt.ylabel('Error', fontsize=label_fontsize)
    plt.title(title, fontsize=title_fontsize)

    plt.savefig(path) if len(path) != 0 else plt.show()
    plt.close()


def plot_pointwise_error(x: List[float], estimated_sol: List[float], analytical_sol: List[float], title: str, path: str,
                         label: str, x_label: str, color: str = "blue"):
    title_fontsize = 17
    label_fontsize = 11
    figsize = (8, 6)

    plt.figure(figsize=figsize)
    error = [abs(e - a) for e, a in zip(estimated_sol, analytical_sol)]
    plt.plot(x, error, color=color, label=label)
    plt.legend()
    plt.xlabel(x_label, fontsize=label_fontsize)
    plt.ylabel('Error', fontsize=label_fontsize)
    plt.title(title, fontsize=title_fontsize)
    plt.savefig(path) if len(path) != 0 else plt.show()
    plt.close()


def plot_embedded_error(abs_tol: List[float], errors: List[float], title: str,
                        label: str, x_label: str, path: str, color: str = "blue"):
    title_fontsize = 17
    label_fontsize = 11
    figsize = (8, 6)

    plt.figure(figsize=figsize)
    plt.loglog(abs_tol, errors, color=color, label=label)
    plt.legend()
    plt.xlabel(x_label, fontsize=label_fontsize)
    plt.ylabel('Error', fontsize=label_fontsize)
    plt.title(title, fontsize=title_fontsize)

    plt.savefig(path) if len(path) != 0 else plt.show()
    plt.close()
