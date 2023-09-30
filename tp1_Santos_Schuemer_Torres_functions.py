import numpy as np

# Tren de pulsos
def pulse_train(A, F):
    """
    Returns the pulse train function.
    """
    return lambda t: A * np.sign(np.sin(2 * np.pi * F * t))

def sf_pulse_train(A):
    """
    Returns the pulse train harmonic coefficients.
    """
    a0 = 0
    an = lambda n: 0
    bn = lambda n: -2 * (A / (np.pi * n)) * (np.cos(np.pi * n) - 1)
    return a0, an, bn


# Diente de sierra
def sawtooth(A, F):
    """
    Returns the sawtooth function.
    """
    return lambda t: A * (t * F - np.floor(0.5 + t * F))

def sf_sawtooth(A, F):
    """
    Returns the sawtooth harmonic coefficients.
    """
    a0 = 0
    an = lambda n: -2 * A * F * np.sin(np.pi * n) / (np.pi * n)
    bn = lambda n: -A * np.cos(np.pi * n) / (np.pi * n)
    return a0, an, bn

# Triangular
def triangle(A, F):
    """
    Returns the triangle function.
    """
    return lambda t: 4*A*abs((t*F - np.floor(0.5 + t*F)))-A

def sf_triangle_wave(A):
    """
    Returns the triangle wave harmonic coefficients.
    """
    a0 = 0
    an = lambda n: 4*A*((-1)**n - 1)/(np.pi**2*n**2)
    bn = lambda n: 0
    return a0, an, bn


# Funciones para sintetizar una señal a partir de sus coeficientes de Fourier (en forma trigonométrica y exponencial)

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


def CuadraticMeanError(f: callable, g: callable, t: np.ndarray, discontinuities: np.ndarray = np.array([])) -> float:
    """
    Returns the cuadratic mean error between f and g in the interval t.
    :param f: function
    :param g: function
    :param t: interval
    """
    return np.mean((f(t) - g(t)) ** 2)


def NHarmonics(f: callable, an: callable, bn: callable, a0: float, T: float, epsilon: float, t: np.ndarray) -> int:
    """
    Returns the number of terms N needed to have a square mean error
    less than epsilon.
    :param f: function
    :param t: interval
    :param epsilon: error
    """
    N = 1
    while CuadraticMeanError(f, trigFourierSeries(an, bn, a0, T, N), t) > epsilon:
        N += 1
    return N


def errorVsTerms(f: callable, an: callable, bn: callable, a0: float, T: float, n: list, t: np.ndarray) -> np.ndarray:
    """
    Returns the error between f and its Fourier series
    for each number of terms.
    :param f: function
    :param an: function that returns the coefficient an
    :param bn: function that returns the coefficient bn
    :param a0: constant term
    :param T: period of the function
    :param epsilon: error
    :param N: number of terms
    """
    return np.array([CuadraticMeanError(f, trigFourierSeries(an, bn, a0, T, n), t) for n in n])


def removeDiscontinuity(t: np.ndarray, discontinuity: list) -> np.ndarray:
    """
    Returns the interval t without the discontinuities.
    :param t: interval
    :param discontinuity: list of discontinuities
    """
    percent = np.abs(t[-1] - t[0]) * 1e-1
    newT = np.array([])
    
    t0 = t[0]
    for d in discontinuity:
        newT = np.append(newT, t[np.where((t >= t0) & (t <= d - percent))])
        t0 = d + percent
    newT = np.append(newT, t[np.where(t >= t0)])

    return newT