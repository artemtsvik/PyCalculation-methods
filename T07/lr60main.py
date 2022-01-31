"""Лабораторна робота №6 з МО
Розв'язування крайової задачі для одновимірного рівняння параболічного типу різницевим методом (явна схема)
"""
import matplotlib.pyplot as plt
import numpy as np


def lr6_u0(x):
    """Функція u0(x)"""
    return x * (1 - x)


def lr6_f(x, t):
    """Функція f(x,t)"""
    return (2 - x * (1 - x)) * np.exp(-t)


def lr6_mu1(t):
    """Функція mu1(t)"""
    return 0


def lr6_mu2(t):
    """Функція mu2(t)"""
    return - np.exp(-t)


# параметри задачі
a, b = 0, 1
l, c1, d1, c2, d2 = 1, 1, 0, 0, 1

# параметри різницевої схеми
N = 51
N1 = N - 1
X = np.linspace(a, b, N)
h = X[1] - X[0]


tau = 2e-4
tau2 = 0.5 * tau
M = 250000

# обчислення додаткових констант (з урахуванням крайових умов)
gam = tau / (h * h)
D2 = 2 * h / d2

# формування початкових умов при t=0
Y = lr6_u0(X)
Y0 = Y.copy()

for j in range(M):
    # знаходження наближеного розв'язку Y на черговому шарі t=t(j)
    t = j * tau
    t1 = t + tau2
    yc = Y[0]

    Y[0] = lr6_mu1(t + tau)
    for k in range(1, N1):
        yl = yc
        yc = Y[k]
        Y[k] = yc + gam * (yl - 2 * yc + Y[k + 1]) + tau * lr6_f(X[k], t1)

    yl = yc
    yc = Y[N1]
    yr = yl + D2 * lr6_mu2(t)
    Y[N1] = yc + gam * (yl - 2 * yc + yr) + tau * lr6_f(l, t1)

    if (j + 1) * 2 == M:
        Y1 = Y.copy()  # запам'ятовуєм для побудови графіку розв'язку в t=M*tau/2


# побудова графіків розв'язку при t=0, M*tau/2, M*tau
t = M * tau
T1 = t/2
T = M * tau

plt.figure(1)
plt.xlabel("x")
plt.ylabel("u")
plt.axis((a, b, None, None))

plt.plot(X, Y0, 'b.-', markevery=6)
plt.plot(X, Y1, 'k.--', markevery=6)
plt.plot(X, Y, 'r*-', markevery=6)

plt.legend(('u(x, 0)', 'u(x,{})'.format(T1), 'u(x, {})'.format(T)))

# порівняння точного розв'язку U(x,t)=x(1-x)exp(-t) і наближеного Y в точці t=M*tau

u = lambda x, t: x * (1 - x) * np.exp(-t)
U = u(X, t)

print("Похибка в нормі C:", np.linalg.norm(Y - U, np.inf))

plt.figure(2)
plt.title("Абсолютна похибка")
plt.xlabel("x")
plt.ylabel("$|u - U|$")
plt.axis((a, b, None, None))
plt.grid(True)

plt.plot(X, np.abs(Y - U), 'r')

plt.show()

