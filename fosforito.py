import numpy as np
import matplotlib.pyplot as plt

A = 1
f = 1

# Pulse wave signal
xt = lambda t: A * np.sign(np.sin(2 * np.pi * f * t))

# Sawtooth wave signal
yt = lambda t: A * (t * f - np.floor(0.5 + t * f))

def sinc(x):
    return 1 if x == 0 else np.sin(np.pi * x) / (np.pi * x)


# Coefficient of the Fourier series
an = lambda n: 0
bn = lambda n: 2 * A / (np.pi * n) * (1 - np.cos(np.pi * n))

# Fourier series
ft = lambda t, n: an(n) + bn(n) * np.sin(2 * np.pi * n * f * t)

# Fourier series approximation
ft_approx = lambda t, n: sum([ft(t, i) for i in range(1, n + 1)])


# Plot the signal and its Fourier series approximation
def plot_signal_and_approximation(signal, approximation, n):
    t = np.arange(-1, 1, 0.01)
    plt.plot(t, signal(t), label='Signal')
    plt.plot(t, approximation(t, n), label='Approximation')
    plt.legend()
    plt.show()

plot_signal_and_approximation(xt, ft_approx, 1000)

