"""Лабораторна робота №6 з МО
Розв'язування крайової задачі для одновимірного рівняння параболічного типу різницевим методом (неявна схема)
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg


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

tau = 1e-2
tau2 = 0.5 * tau
M = 2500
sigma = 0.5

# обчислення додаткових констант (з урахуванням крайових умов)
gam = tau / (h * h)
gs = gam * sigma
gs2 = 2 * gs
s1 = 1 - sigma
gs1 = gam * s1

ck = 1 + gs2
ckj = 1 - 2 * gs1
D2 = 2 * h / d2
D2gs = gs * D2

# формування початкових умов при t=0 і матриці СЛАР
Y = lr6_u0(X)
A = np.full((3, N), [[gs], [-ck], [gs]])

A[1, 0], A[0, 1] = -1, 0  # враховуєм крайові умови при x=0
A[2, -2] = gs2  # враховуєм крайові умови при x=l

Y0 = Y.copy()  # запам'ятовуєм для побудови графіку розв'язку в t=0
f = np.zeros(N)  # права частина СЛАР

for j in range(M):
    # знаходження наближеного розв'язку Y на черговому шарі t=t(j)
    t = (j + 1) * tau
    t1 = t - tau2

    # формування правої частину СЛАР
    f[0] = -lr6_mu1(t)
    for k in range(1, N1):
        f[k] = - gs1 * (Y[k-1] + Y[k+1]) - ckj * Y[k] - tau * lr6_f(X[k], t1)
    f[N1] = - gs1 * (2*Y[N1-1] + D2*lr6_mu2(t - tau)) - ckj * Y[N1] - tau * lr6_f(l, t1) - D2gs * lr6_mu2(t)

    # розв'язок СЛАР
    Y[:] = scipy.linalg.solve_banded((1, 1), A, f, overwrite_b=True, check_finite=False)

    if (j + 1) * 2 == M:
        Y1 = Y.copy()  # запам'ятовуємо для побудови графіка розв'язку при t=M*tau/2


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

plt.legend(('u(x, 0)', 'u(x, {})'.format(T1), 'u(x, {})'.format(T)))

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

