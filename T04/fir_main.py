"""Тест u(x) = x**3 * (1-x),
u" = 6x(1-2x), x є (0,1),
u(0) = 0, u(1) = 0.
"""
import matplotlib.pyplot as plt
import numpy as np

from ode import firlin

# вхідні дані
fir_a = lambda x: np.array([[0., 1.], [0., 0.]])
fir_f = lambda x: np.array([[0.], [6 * x * (1 - 2 * x)]])

a, b = 0, 1
D = np.array([1., 0., 0., 1., 0., 0.])

# формування вузлiв виведення результатів
n = 51
X = np.linspace(a, b, n)
U = firlin(X, D, fir_a, fir_f)

# порівняння наближеного і точного розв'язку
u_true = lambda x: x**3 * (1 - x)
du = lambda x: x * x * (3 - 4 * x)

plt.xlabel("x")
plt.ylabel("$u(x), u^{\prime}(x)$")
plt.axis((a, b, -2.5, 2.5))
plt.title("Розв'язання КЗ для ЗДР балістичним методом")

plt.plot(X, U[0], X, u_true(X), '+', X, U[1], X, du(X), '*')

plt.legend(('u(x)', '$u_{true}(x)$', '$u^{\prime}(x)$', '$u_{true}^{\prime}(x)$'))

plt.show()

