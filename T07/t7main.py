"""Розв'язування квазілінійної крайової задачі для одновимірного
рівняння параболічного типу різницевим методом (симетрична схема)
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg


def t7_u0(x):
    """Функція u0(x)"""
    return np.tan(x)


def t7_f(u):
    """Функція f(u)"""
    return (1 + u * u) * (1 - 2 * u)


def t7_dfu(u):
    """Похідна від функції f(u) по u"""
    return 2 * u * (1 - 3 * u) - 2


def t7_mu1(t):
    """Функція mu1(t)"""
    global a
    return np.tan(a + t)


def t7_mu2(t):
    """Функція mu2(t)"""
    global b
    return np.tan(b + t)


# параметри задачі
a, b = 0, 1

# параметри різницевої схеми та ітераційного процесу
N = 201
N1 = N - 1
X = np.linspace(a, b, N)
h = X[1] - X[0]

tau = 1e-3
M = 500
eps = 1e-8
smax = 1000

# обчислення додаткових констант
tau2 = 0.5 * tau
gs = tau2 / (h * h)
gs2 = 2 * gs
ck = 1 + gs2
ckj = 1 - gs2

# формування початкових умов при t=0 і матриці СЛАР
A = np.full((3, N), [[gs], [0], [gs]])  # np.zeros((3, N))
f = np.zeros(N)

fj = f.copy()
Y = t7_u0(X)

A[1, 0], A[0, 1] = -1, 0  # враховуємо крайові умови при x=a
A[2, -2], A[1, -1] = 0, -1  # враховуємо крайові умови при x=b

Y0 = Y.copy()
Yj = np.zeros(N)
Ys = Yj.copy()

for j in range(M):
    # знаходження наближеного розв'язку Y на черговому шарі t=t(j)
    t = (j + 1) * tau
    # формування фіксованої правої частини
    for k in range(1, N1):
        fj[k] = gs * (Y[k-1] + Y[k+1]) + ckj * Y[k]
    f[0] = - t7_mu1(t)
    f[-1] = - t7_mu2(t)

    # ітераційний процес на j-му шарі
    Yj[:] = Y
    Ys = np.zeros(N)
    for s in range(smax):
        Ys[:] = Y
        # формування СЛАР
        for k in range(1, N1):
            y = Ys[k]
            yy = 0.5 * (y + Yj[k])
            gu = tau2 * t7_dfu(yy)
            A[1, k] = gu - ck
            f[k] = gu * y - fj[k] - tau * t7_f(yy)

        # розв'язок СЛАР, знаходження Y
        Y[:] = scipy.linalg.solve_banded((1, 1), A, f, check_finite=False)

        if np.all(np.abs(Y - Ys) < eps):
            break

    if (j + 1) * 2 == M:
        Y1 = Y.copy()


# побудова графіків розв'язку при t=0, M*tau/2, M*tau
t = M * tau
T1 = t/2
T = M * tau

plt.figure(1)
plt.title("Наближений розв'язок квазілінійної КЗ")
plt.xlabel("x")
plt.ylabel("u")
plt.axis((a, b, None, None))

plt.plot(X, Y0, 'b.-', markevery=6)
plt.plot(X, Y1, 'k.--', markevery=6)
plt.plot(X, Y, 'r*-', markevery=6)

plt.legend(('u(x, 0)', 'u(x, {})'.format(T1), 'u(x, {})'.format(T)))

# порівняння точного розв'язку U(x,t)=tg(x+t) і наближеного Y в точці t=M*tau

u = lambda x, t: np.tan(x + t)
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

