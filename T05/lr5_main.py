"""Лабораторна робота №5 з МО
Розв'язування крайової задачі для нестаціонарних рівнянь та систем рівнянь різних типів методом Роте (прямих)
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


def lr5_u0(X):
    """Функції U(x,0), V(x,0)"""
    uv = np.apply_along_axis(lambda x: [np.sin(np.pi * x), np.cos(np.pi * x)], 0, X)
    uv.shape = -1
    return uv


def lr5_my1(t):
    """Функція U(0, t)"""
    return np.sin(-t)


def lr5_my2(t):
    """Функція V(1,t)"""
    return np.cos(np.pi - t)


def lr5_ff(t, U):
    """Опис правої частини системи звичайних диференціальних рівнянь"""
    global k, g, D, N, N1, NN, X
    n = NN - 2
    ii = N1 - 1
    f = np.zeros(n)

    yy1, yyNN = lr5_my1(t), lr5_my2(t)
    f[0] = np.pi * np.cos(np.pi * X[1] - t) - g * U[N] - D * (U[0] - yy1)

    for i in range(1, ii):
        i1, i2, i3 = N1 + i, N + i, i - 1
        f[i] = np.pi * np.cos(np.pi * X[i + 1] - t) - g * U[i2] - D * (U[i] - U[i3])
        f[i1] = D * (U[i2] - U[i1]) + k * U[i3] + np.pi * np.sin(np.pi * X[i] - t)

    f[ii] = np.pi * np.cos(np.pi * X[-1] - t) - g * yyNN - D * (U[ii] - U[i])
    f[N1] = D * (U[N] - U[N1]) + k * yy1 + np.pi * np.sin(np.pi * X[0] - t)
    f[-1] = D * (yyNN - U[-1]) + k * U[i] + np.pi * np.sin(np.pi * X[ii] - t)

    return f


# параметри модельної задачі
a, b = 0, 1
k, g, d = 1, 1, 1

# параметри різницевої схеми
N = 51
X = np.linspace(a, b, N)
h = X[1] - X[0]

N1 = N - 1
NN = 2 * N
NN1 = NN - 1

tau = 1e-2
M = 1000
D = 1 / (d * h)

# формування початкових умов при t=0
Y = lr5_u0(X)
Y0 = Y.copy()  # запам'ятовуємо для побудови графіка розв'язку при t=0
tk = 0

for j in range(M):
    t0 = tk
    tk += tau

    sol = solve_ivp(lr5_ff, (t0, tk), Y[1:-1], method='RK45')
    y = sol.y[:, -1]

    Y[0] = lr5_my1(tk)
    for i, yi in enumerate(y):
        Y[i+1] = yi
    Y[-1] = lr5_my2(tk)

    if (j + 1) * 2 == M:
        Y1 = Y.copy()  # запам'ятовуємо для побудови графіка розв'язку при t=M*tau/2


# побудова графіків розв'язку при t=0, M*tau/2, M*tau
t = M * tau
t1 = 0.5 * t

plt.figure(1)
plt.xlabel("x")
plt.ylabel("u")
plt.axis((a, b, None, None))
plt.plot(X, Y0[:N], 'b.-', markevery=6)
plt.plot(X, Y1[:N], 'ko--', markevery=6)
plt.plot(X, Y[:N], 'r*-', markevery=6)

plt.legend(('u(x, 0)', 'u(x, {})'.format(t1), 'u(x, {})'.format(t)))

plt.figure(2)
plt.xlabel("x")
plt.ylabel("v")
plt.axis((a, b, None, None))
plt.plot(X, Y0[N:], 'b.-', markevery=6)
plt.plot(X, Y1[N:], 'ko--', markevery=6)
plt.plot(X, Y[N:], 'r*-', markevery=6)

plt.legend(('v(x, 0)', 'v(x, {})'.format(t1), 'v(x, {})'.format(t)))

# порівняння точного розв'язку
# U(x,t)=sin(pi*x-t), V(x,t)=cos(pi*x-t)
# і наближеного Y(:) у точці t=M*tau
u = lambda x, t: np.sin(np.pi * x - t)
U = u(X, t)

plt.figure(3)
plt.xlabel("x")
plt.ylabel("u")
plt.axis((a, b, None, None))
plt.plot(X, Y[:N], 'b.-', markevery=6)
plt.plot(X, U, 'r*-', markevery=6)

plt.legend(('u(x, {})'.format(t), 'U(x, {})'.format(t)))

print("Похибка для u(x, {}) в нормі C:".format(t), np.linalg.norm(Y[:N] - U, np.inf))

v = lambda x, t: np.cos(np.pi * x - t)
V = v(X, t)

plt.figure(4)
plt.xlabel("x")
plt.ylabel("v")
plt.axis((a, b, None, None))
plt.plot(X, Y[N:], 'b.-', markevery=6)
plt.plot(X, V, 'r*-', markevery=6)

plt.legend(('v(x, {})'.format(t), 'V(x, {})'.format(t)))

print("Похибка для v(x, {}) в нормі C:".format(t), np.linalg.norm(Y[N:] - V, np.inf))

plt.show()

