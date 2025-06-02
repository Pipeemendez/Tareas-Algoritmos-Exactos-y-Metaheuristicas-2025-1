#APUNTES CLASE
""" DE (Diferential Evolution):
        Generar poblacion inicial de soluciones P0 de tamaño k. Cada individuo corresponde a un vectorr real Xij y de dimension D.

        Elementos/parámetros: Jrand, F (0,8) y CR (0,9)
"""

import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import time
import math

# f1(x) = 4 - 4x1^3 - 4x1 + x2^2
# Dominio: -5 <= xi <= 5 para i=1, 2
def f1(x):
    if len(x) != 2:
        raise ValueError("f1 espera una entrada de 2 dimensiones.")
    x1, x2 = x
    return 4 - 4 * x1**3 - 4 * x1 + x2**2

# f2(x) = 1/899 * sum(x_i^2 - 1745) para i=1 a 6
# Dominio: 0 <= xi <= 1 para i=1 a 6
def f2(x):
    if len(x) != 6:
        raise ValueError("f2 espera una entrada de 6 dimensiones.")
    return (1/899.0) * np.sum(x**2 - 1745)

# f3(x) = (x1^4 + x2^4 - 17)^2 + (2x1 + x2 - 4)^2
# Dominio: -500 <= xi <= 500 para i=1, 2
def f3(x):
    if len(x) != 2:
        raise ValueError("f3 espera una entrada de 2 dimensiones.")
    x1, x2 = x
    return (x1**4 + x2**4 - 17)**2 + (2*x1 + x2 - 4)**2

# f4(x) = sumatoria((ln(x_i - 2))^2 + (ln(10 - x_i))^2) - productoria(x_i)^0.2 para i=1 a 10
# Dominio: -2.001 < xi <= 10 para i=1 a 10
def f4(x):
    if len(x) != 10:
        raise ValueError("f4 espera una entrada de 10 dimensiones.")

    # penalización
    if np.any(x <= 2):
        return 1e10

    if np.any(x >= 10):
        return 1e10

    term_sum = np.sum((np.log(x - 2))**2 + (np.log(10 - x))**2)

    # Calcular el término de la productoria.
    try:
        term_product = np.prod(x)
        term_prod_power = np.power(term_product, 0.2)
    except Exception as e:
        print(f"Advertencia: Error calculando el término de productoria en f4: {e}")
        return 1e10

    return term_sum - term_prod_power