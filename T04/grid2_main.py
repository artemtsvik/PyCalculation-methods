"""Розв'язання модельної задачі
    U"(x) - p(x)U(x) = -f(x),
    p(x)>=0, x Є (0,1);
    U'(0) + U(0) = 0,
    U'(1) + U(1) = -1.
Для p(x) = 5*(2-x), f(x) = x*(6*(2*x-1)+5*(1-x)*(2-x)x**2) розв'язок буде U(x) = (1-x)*x**3.
"""
import matplotlib.pyplot as plt
import numpy as np

from ode import m_grid2


# Вхідні дані
a, b = 0, 1
D = np.array([1., 1., 0., 1., 1., -1.])

pfun = lambda x: 5 * (2 - x)
ffun = lambda x: (x * (6 * (2 * x - 1) + 5 * (1 - x) * (2 - x) * x * x))

# Модель
u = lambda x: (1 - x) * x**3

# Наближений розв'язок
n = 51
X, Y = m_grid2(pfun, ffun, a, b, D, n)

# Точний розв'язок
U = u(X)

plt.figure(1)
plt.title("Різницевий метод 2-го порядку")
plt.xlabel("x")
plt.ylabel("y")
plt.axis((a, b, None, None))

plt.plot(X, Y)

print("Похибка в нормі C=", np.linalg.norm(Y - U))

plt.figure(2)
plt.title("Абсолютна похибка")
plt.xlabel("x")
plt.ylabel("$\left|y - u(x)\\right|$")
plt.axis((a, b, None, None))
plt.grid(True)

plt.plot(X, np.abs(U - Y), 'r')

plt.show()

