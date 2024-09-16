import numpy as np
from scipy.fft import fft, ifft
from scipy.integrate import quad
from scipy.stats import norm
import matplotlib.pyplot as plt

# ---------------------------
# Option Pricing Models
# ---------------------------

def characteristic_function(u, S0, K, T, r, sigma):
    """
    Computes the characteristic function for the underlying asset's log returns.

    Parameters:
    - u: Integration variable
    - S0: Current stock price
    - K: Strike price
    - T: Time to maturity
    - r: Risk-free interest rate
    - sigma: Volatility

    Returns:
    - Characteristic function value
    """
    return np.exp(1j * u * (np.log(S0) + (r - 0.5 * sigma**2) * T) - 0.5 * sigma**2 * u**2 * T)

def option_price_fft(S0, K, T, r, sigma, N=4096, alpha=1.5):
    """
    Prices a European call option using the Fast Fourier Transform (FFT) method.

    Parameters:
    - S0: Current stock price
    - K: Strike price
    - T: Time to maturity
    - r: Risk-free interest rate
    - sigma: Volatility
    - N: Number of FFT points
    - alpha: Damping factor

    Returns:
    - Option price calculated using FFT
    """
    # Define the integration range and step size
    eta = 0.25
    lam = 2 * np.pi / (N * eta)
    b = N * lam / 2
    k = -b + lam * np.arange(N)

    # Compute the characteristic function
    u = np.arange(N) * eta
    cf = characteristic_function(u - (alpha + 1) * 1j, S0, K, T, r, sigma)

    # Apply damping factor
    cf = cf * np.exp(-r * T)

    # Simpson's rule weights
    Simpson_weights = np.ones(N)
    Simpson_weights[1:N - 1:2] = 4
    Simpson_weights[2:N - 2:2] = 2

    # Compute the FFT
    integrand = np.exp(-1j * u * b) * cf * Simpson_weights
    fft_values = fft(integrand).real

    # Compute the option price
    C = np.exp(-alpha * k) / np.pi * fft_values
    # Interpolate to find the price corresponding to the desired strike K
    price_index = np.argmin(np.abs(k - np.log(K / S0)))
    return C[price_index]

def option_price_quadrature(S0, K, T, r, sigma):
    """
    Prices a European call option using numerical quadrature.

    Parameters:
    - S0: Current stock price
    - K: Strike price
    - T: Time to maturity
    - r: Risk-free interest rate
    - sigma: Volatility

    Returns:
    - Option price calculated using Quadrature
    """
    # Define the integrand for the Black-Scholes formula
    def integrand(x):
        return np.exp(-0.5 * x ** 2) * (S0 * np.exp(x * sigma * np.sqrt(T)) - K) * norm.cdf(x)

    # Perform numerical integration
    price, error = quad(integrand, -np.inf, np.inf)
    return price * np.exp(-r * T)

def black_scholes_price(S0, K, T, r, sigma):
    """
    Prices a European call option using the Black-Scholes formula.

    Parameters:
    - S0: Current stock price
    - K: Strike price
    - T: Time to maturity
    - r: Risk-free interest rate
    - sigma: Volatility

    Returns:
    - Option price calculated using Black-Scholes
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    C = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return C

def combined_option_price(S0, K, T, r, sigma):
    """
    Combines FFT and Quadrature option prices by averaging them.

    Parameters:
    - S0: Current stock price
    - K: Strike price
    - T: Time to maturity
    - r: Risk-free interest rate
    - sigma: Volatility

    Returns:
    - Combined option price
    """
    fft_price = option_price_fft(S0, K, T, r, sigma)
    quad_price = option_price_quadrature(S0, K, T, r, sigma)
    # Combine the prices (average)
    combined_price = (fft_price + quad_price) / 2
    return combined_price

# ---------------------------
# Parameters and Calculations
# ---------------------------

# Sample Parameters
S0 = 100    # Current stock price
K = 100     # Strike price
T = 1       # Time to maturity in years
r = 0.05    # Risk-free interest rate
sigma = 0.2 # Volatility

# Strike Prices Range for Comparison
K_values = np.linspace(80, 120, 100)

# Calculate Option Prices Using Different Methods
prices_fft = [option_price_fft(S0, K_val, T, r, sigma) for K_val in K_values]
prices_quad = [option_price_quadrature(S0, K_val, T, r, sigma) for K_val in K_values]
prices_bs = [black_scholes_price(S0, K_val, T, r, sigma) for K_val in K_values]
prices_combined = [combined_option_price(S0, K_val, T, r, sigma) for K_val in K_values]

# Calculate Accuracy Metrics
# Assuming Black-Scholes as the reference (since it's analytical)
errors_fft = [abs((fft - bs) / bs) * 100 for fft, bs in zip(prices_fft, prices_bs)]
errors_quad = [abs((quad - bs) / bs) * 100 for quad, bs in zip(prices_quad, prices_bs)]
errors_combined = [abs((comb - bs) / bs) * 100 for comb, bs in zip(prices_combined, prices_bs)]

# ---------------------------
# Plotting the Results
# ---------------------------

plt.figure(figsize=(14, 8))

# Plot Option Prices
plt.subplot(2, 1, 1)
plt.plot(K_values, prices_bs, label='Black-Scholes Price', color='black', linewidth=2)
plt.plot(K_values, prices_fft, label='FFT-Based Price', linestyle='--')
plt.plot(K_values, prices_quad, label='Quadrature-Based Price', linestyle='-.')
plt.plot(K_values, prices_combined, label='Combined FFT & Quadrature Price', linewidth=2)
plt.xlabel('Strike Price (K)')
plt.ylabel('Option Price')
plt.title('Option Pricing Comparison')
plt.legend()
plt.grid(True)

# Plot Accuracy Metrics
plt.subplot(2, 1, 2)
plt.plot(K_values, errors_fft, label='FFT Error (%)', linestyle='--')
plt.plot(K_values, errors_quad, label='Quadrature Error (%)', linestyle='-.')
plt.plot(K_values, errors_combined, label='Combined Method Error (%)', linewidth=2)
plt.xlabel('Strike Price (K)')
plt.ylabel('Absolute Percentage Error (%)')
plt.title('Accuracy Comparison (Relative to Black-Scholes)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ---------------------------
# Output Prices and Accuracy
# ---------------------------

# Calculate Option Prices for a Specific Strike (K=100)
fft_price = option_price_fft(S0, K, T, r, sigma)
quad_price = option_price_quadrature(S0, K, T, r, sigma)
bs_price = black_scholes_price(S0, K, T, r, sigma)
combined_price = combined_option_price(S0, K, T, r, sigma)

# Calculate Errors
error_fft = abs((fft_price - bs_price) / bs_price) * 100
error_quad = abs((quad_price - bs_price) / bs_price) * 100
error_combined = abs((combined_price - bs_price) / bs_price) * 100

print(f"----- Option Pricing Comparison at K={K} -----")
print(f"Black-Scholes Price: {bs_price:.4f}")
print(f"FFT-Based Price: {fft_price:.4f}")
print(f"Quadrature-Based Price: {quad_price:.4f}")
print(f"Combined FFT & Quadrature Price: {combined_price:.4f}\n")

print(f"----- Accuracy (Relative to Black-Scholes) -----")
print(f"FFT Error: {error_fft:.2f}%")
print(f"Quadrature Error: {error_quad:.2f}%")
print(f"Combined Method Error: {error_combined:.2f}%")
