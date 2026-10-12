"""
Black-Scholes Option Pricing Calculator
---------------------------------------
1. Prices European call/put options with the Black-Scholes formula
2. Verifies results using put-call parity
3. Prices the same options by Monte Carlo simulation of Geometric Brownian Motion
4. Plots how option prices change with volatility, strike and time to expiry
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# ---------- 1. Black-Scholes formula ----------
def black_scholes(S, K, T, r, sigma, option="call"):
    """
    S     : current stock price
    K     : strike price
    T     : time to expiry (years)
    r     : risk-free interest rate (annual, continuous)
    sigma : volatility (annual)
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ---------- 2. Put-call parity check ----------
def check_put_call_parity(S, K, T, r, sigma):
    """Put-call parity: C - P = S - K * e^(-rT)"""
    C = black_scholes(S, K, T, r, sigma, "call")
    P = black_scholes(S, K, T, r, sigma, "put")
    lhs = C - P
    rhs = S - K * np.exp(-r * T)
    return lhs, rhs, abs(lhs - rhs)


# ---------- 3. Monte Carlo pricing ----------
def simulate_gbm_paths(S, T, r, sigma, n_steps=252, n_paths=10, seed=42):
    """Simulate full stock price paths (used for plotting)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(increments, axis=1)
    paths = S * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))
    return paths


def monte_carlo_price(S, K, T, r, sigma, option="call", n_sims=200_000, seed=42):
    """
    Under GBM the final price is:
        S_T = S * exp((r - sigma^2/2) T + sigma sqrt(T) Z),  Z ~ N(0,1)
    Option price = discounted average payoff.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_sims)
    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    if option == "call":
        payoff = np.maximum(S_T - K, 0)
    else:
        payoff = np.maximum(K - S_T, 0)
    price = np.exp(-r * T) * payoff.mean()
    std_error = np.exp(-r * T) * payoff.std() / np.sqrt(n_sims)
    return price, std_error


# ---------- 4. Plots ----------
def make_plots(S, K, T, r, sigma):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) Price vs volatility
    vols = np.linspace(0.05, 0.8, 100)
    axes[0, 0].plot(vols, black_scholes(S, K, T, r, vols, "call"), label="Call")
    axes[0, 0].plot(vols, black_scholes(S, K, T, r, vols, "put"), label="Put")
    axes[0, 0].set(title="Option Price vs Volatility", xlabel="Volatility (σ)", ylabel="Price")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # (b) Price vs strike
    strikes = np.linspace(0.5 * S, 1.5 * S, 100)
    axes[0, 1].plot(strikes, black_scholes(S, strikes, T, r, sigma, "call"), label="Call")
    axes[0, 1].plot(strikes, black_scholes(S, strikes, T, r, sigma, "put"), label="Put")
    axes[0, 1].axvline(S, color="gray", ls="--", label="Spot price")
    axes[0, 1].set(title="Option Price vs Strike", xlabel="Strike (K)", ylabel="Price")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # (c) Price vs time to expiry
    times = np.linspace(0.01, 2, 100)
    axes[1, 0].plot(times, black_scholes(S, K, times, r, sigma, "call"), label="Call")
    axes[1, 0].plot(times, black_scholes(S, K, times, r, sigma, "put"), label="Put")
    axes[1, 0].set(title="Option Price vs Time to Expiry", xlabel="Time (years)", ylabel="Price")
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # (d) Sample GBM paths
    paths = simulate_gbm_paths(S, T, r, sigma, n_paths=15)
    t = np.linspace(0, T, paths.shape[1])
    for p in paths:
        axes[1, 1].plot(t, p, lw=0.8)
    axes[1, 1].axhline(K, color="black", ls="--", label="Strike")
    axes[1, 1].set(title="Simulated Stock Paths (GBM)", xlabel="Time (years)", ylabel="Stock price")
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("option_pricing_plots.png", dpi=150)
    print("\nPlots saved to option_pricing_plots.png")


# ---------- Main ----------
if __name__ == "__main__":
    # Example inputs (edit freely)
    S = 100      # stock price
    K = 100      # strike
    T = 1.0      # 1 year
    r = 0.05     # 5% risk-free rate
    sigma = 0.20 # 20% volatility

    print("=" * 55)
    print("  Black-Scholes Option Pricing Calculator")
    print("=" * 55)
    print(f"S={S}, K={K}, T={T}, r={r}, sigma={sigma}\n")

    bs_call = black_scholes(S, K, T, r, sigma, "call")
    bs_put = black_scholes(S, K, T, r, sigma, "put")
    print(f"Black-Scholes Call : {bs_call:.4f}")
    print(f"Black-Scholes Put  : {bs_put:.4f}")

    lhs, rhs, diff = check_put_call_parity(S, K, T, r, sigma)
    print(f"\nPut-call parity:  C - P = {lhs:.4f},  S - Ke^(-rT) = {rhs:.4f},  diff = {diff:.2e}")

    mc_call, se_call = monte_carlo_price(S, K, T, r, sigma, "call")
    mc_put, se_put = monte_carlo_price(S, K, T, r, sigma, "put")
    print(f"\nMonte Carlo Call   : {mc_call:.4f}  (± {se_call:.4f})")
    print(f"Monte Carlo Put    : {mc_put:.4f}  (± {se_put:.4f})")
    print(f"Call error vs BS   : {abs(mc_call - bs_call) / bs_call * 100:.3f}%")
    print(f"Put error vs BS    : {abs(mc_put - bs_put) / bs_put * 100:.3f}%")

    make_plots(S, K, T, r, sigma)
