"""Розв'язування задачі
    U"xx + U"yy = -f(x,y), (x,y) є П,
    U(x,y) = mu(x,y), (x,y) є dП.
П U dП = (0 <= x <= L(1) & 0 <= y <= L(2))
"""
import matplotlib.pyplot as plt
import numpy as np

from dppde import mznopj, ptmopch


def t82_f(x1, x2):
    """Функція f(x1,x2)"""
    return 6 * (x1 + x2) - 6


def t82_mux(x):
    """Функція mu(x,y) для y = 0, L[1]"""
    z = x * x * (1 - x)
    return np.array([z, z])


def t82_muy(y):
    """Функція mu(x,y) для x = 0, L[0]"""
    z = y * y * (2 - y)
    return np.array([z, z])


# параметри задачі і точність ІП
L = np.array([1, 2])
N = np.array([21, 41])
eps = 1e-8

#  п.1 ІП - метод змінних напрямків (МЗН) з вибором оптимальних параметрів (ОП) по Жордану
n, X, Y, U1 = mznopj(L, N, eps, t82_f, t82_mux, t82_muy)

# побудова графіка наближеного розв'язку
Yy, Xx = np.meshgrid(Y, X)

plt.figure(1)
ax = plt.axes(projection='3d')
ax.set_title("Метод змінних напрямків")
ax.plot_surface(Xx, Yy, U1, cmap='viridis')

# порівняння точного розв'язку U(x,y)=(1-x)*x**2+(2-y)*y**2 і наближеного Y
u = lambda x, y: x * x * (1 - x) + y * y * (2 - y)
UU = u(Xx, Yy)

U_eps = np.abs(U1 - UU)

plt.figure(2)
ax = plt.axes(projection='3d')
ax.set_title("Абсолютна похибка (МЗН)")
ax.plot_surface(Xx, Yy, U_eps, cmap='hot')


print("Сіткові кроки:\n\tHx =", X[1] - X[0], "\n\tHy =", Y[1] - Y[0])
print("Точність ІП (eps) = ", eps)
print("МЗН з ОП по Жордану: ітерацій =", n)
print("Похибка: ||Y-U||c =", np.max(U_eps))

# п.2 ІП - позмінно-трикутний метод (ПТМ) з чебишовським набором параметрів (ЧНП)
n, X, Y, U2 = ptmopch(L, N, eps, t82_f, t82_mux, t82_muy)

plt.figure(3)
ax = plt.axes(projection='3d')
ax.set_title("Позмінно-трикутний метод")
ax.plot_surface(Xx, Yy, U2, cmap='viridis')

U_eps1 = np.abs(U2 - UU)

plt.figure(4)
ax = plt.axes(projection='3d')
ax.set_title("Абсолютна похибка (ПТМ)")
ax.plot_surface(Xx, Yy, U_eps1, cmap='hot')

print("ПТМ з ЧНП: ітерацій =", n)
print("Похибка: ||Y-U||c =", np.max(U_eps1))


plt.show()

