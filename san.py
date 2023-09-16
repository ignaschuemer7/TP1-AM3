import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

plt.style.use('tp1.mplstyle')

A = 1
F = 1

# Pulse wave
x = lambda t: A * np.sign(np.sin(2 * np.pi * F * t))

# Sawtooth wave
y = lambda t: A * (t * F - np.floor(0.5 + t * F))


def trigFourierSeries(an: callable, bn: callable, a0: float, T: float, N: int) -> callable:
    """
    Returns the trigonometric Fourier series of the function given
    the coefficients an and bn and the number of terms N.
    :param an: function that returns the coefficient an
    :param bn: function that returns the coefficient bn
    :param a0: constant term
    :param T: period of the function
    :param N: number of terms
    """
    def f(t: np.ndarray) -> np.ndarray:
        return a0 / 2 + sum(an(n) * np.cos(2 * np.pi * n * t / T) + bn(n) * np.sin(2 * np.pi * n * t / T) for n in range(1, N + 1))
    return f

def expFourierSeries(cn: callable, c0: float, T: float, N: int) -> callable:
    """
    Returns the exponential Fourier series of the function given
    the coefficient cn and the number of terms N.
    :param cn: function that returns the coefficient cn
    :param c0: constant term
    :param T: period of the function
    :param N: number of terms
    """
    def f(t: np.ndarray) -> np.ndarray:
        return c0 + sum(cn(n) * np.exp(2 * np.pi * 1j * n * t / T) for n in range(-N, N + 1) if n != 0)
    return f

a0 = 0
an = lambda n: 0
bn = lambda n: -2 * (A / (np.pi * n)) * (np.cos(np.pi * n) - 1)
c0 = 0
cn = lambda n: 1j * (A / (np.pi * n)) * (np.cos(np.pi * n) - 1)
T = 1 / F
N = 100


# Valores de t
t = np.linspace(-1, 1, 1000)

# Crear la figura y los ejes para el gráfico principal y el gráfico de zoom
fig, ax = plt.subplots(figsize=(20, 12))

# Plotea la función normal y las aproximaciones en el gráfico principal
ax.plot(t, x(t), label='x(t)')
ax.plot(t, trigFourierSeries(an, bn, a0, T, 10)(t), label='Fourier series with 10 terms')
ax.plot(t, trigFourierSeries(an, bn, a0, T, 30)(t), label='Fourier series with 30 terms')
ax.plot(t, trigFourierSeries(an, bn, a0, T, 50)(t), label='Fourier series with 50 terms')

# Configurar etiquetas y leyendas en el gráfico principal
ax.set_xlabel('t')
ax.set_ylabel('x(t)')
ax.set_title('Pulse wave and its Fourier approximation')
ax.legend(loc='lower left', fontsize="7")

# Establecer los límites originales en el gráfico principal
original_xlim = ax.get_xlim()
original_ylim = ax.get_ylim()

# Establecer los límites en el gráfico principal para mostrar el rectángulo
ax.set_xlim(original_xlim)
ax.set_ylim((-1.25, 1.25))

# Crear un eje de zoom (sub-gráfico) en el mismo figure
axins = inset_axes(ax, width="20%", height="20%", loc='upper right', bbox_to_anchor=(0.07, 0.07, 0.9, 0.9), bbox_transform=ax.transAxes)
axins.plot(t, x(t))
axins.plot(t, trigFourierSeries(an, bn, a0, T, 10)(t))
axins.plot(t, trigFourierSeries(an, bn, a0, T, 30)(t))
axins.plot(t, trigFourierSeries(an, bn, a0, T, 50)(t))

# Establecer los límites en el gráfico de zoom
axins.set_xlim(-0.025, 0.1)
axins.set_ylim(0.85, 1.22)

# Dibujar un rectángulo en el gráfico principal para indicar el área de zoom
mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec="0.5")

# Mostrar la figura completa
plt.show()

