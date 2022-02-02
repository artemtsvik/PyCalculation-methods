"""Розв'язування крайової задачі для одновимірного рівняння гіперболічного типу 2-го порядку
різницевим методом (sigm є (0, 0.5] - неявна схема)
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg


# параметри задачі
ffun = lambda x, t: np.exp(-t) * (2*(3*x - 1)+x*x*(1 - x))  # функція f(x,t)
u0 = lambda x: x * x * (1 - x)                              # початкова умова u(x,0) = u0(x)
u0dx2 = lambda x: 2 - 6*x                                   # 2-га похідна по x від u0(x)
up0 = lambda x: x * x * (x - 1)                             # початкова умова u't(x,0) = up0(x)
g1 = lambda t: 0                                            # крайова умова в х = a u(a,t) = g1(t)
g2 = lambda t: 0                                            # крайова умова в х = b u(b,t) = g2(t)
a, b = 0, 1

# параметри РЗ
N = 101
X = np.linspace(a, b, N)
h = X[1] - X[0]
N1 = N - 1

tau = 1e-2
M = 500
sigm = 0.25

# обчислення додаткових констант
tau2 = tau * tau
t2 = 0.5 * tau2
si = 1 - 2 * sigm

hs = h / tau
hs *= hs / sigm
cs = 2 + hs

ss = 1 / sigm - 2
c1s = 2 * (hs - ss)
hs = h * h / sigm

# формування коефiцiєнтiв A, C, B, F, k=1,...,N для СЛАР
A = np.full((3, N), [[1], [-cs], [1]])
A[0, 1] = 0
A[1, 0] = A[1, -1] = -1
A[2, -2] = 0

f = np.zeros(N)

Up = np.zeros(N)

# формування початкових умов при t=0
t = 0
Um = u0(X)
U = Um + tau * up0(X) + t2 * (u0dx2(X) + ffun(X, t))

t += tau
U[0] = g1(t)
U[-1] = g2(t)

U0 = Um.copy()  # запам'ятовуєм для побудови графіку розв'язку в t=0

for j in range(1, M):
    # знаходження наближеного розв'язку Y(:) на черговому шарі t=t(j)
    for k in range(1, N1):
        f[k] = - Um[k-1] + cs * Um[k] - Um[k+1] - ss * (U[k-1] + U[k+1]) - c1s * U[k] - hs * ffun(X[k], t)

    t += tau
    f[0] = - g1(t)
    f[-1] = - g2(t)

    Up[:] = scipy.linalg.solve_banded((1, 1), A, f, overwrite_b=True, check_finite=False)

    if (j + 1) * 2 == M:
        U1 = Up.copy()  # запам'ятовуєм для побудови графіку в t=M*tau/2

    Um[:] = U
    U[:] = Up


# побудова графіків розв'язку при t=0, M*tau/2, M*tau
T = M * tau
T1 = t / 2

plt.figure(1)
plt.xlabel("x")
plt.ylabel("u")
plt.axis((a, b, None, None))

plt.plot(X, U0, 'b.-', markevery=6)
plt.plot(X, U1, 'k.--', markevery=6)
plt.plot(X, U, 'r*-', markevery=6)

plt.legend(('u(x, 0)', 'u(x, {})'.format(T1), 'u(x, {})'.format(T)))

# порівняння точного розв'язку U(x,t)=x**2*(1-x)*exp(-t) і наближеного U в точці t=M*tau
u = lambda x, t: x * x * (1 - x) * np.exp(-t)

Um[:] = u(X, t)

print("Похибка в нормі C:", np.linalg.norm(U - Um, np.inf))

plt.figure(2)
plt.title("Абсолютна похибка")
plt.xlabel("x")
plt.ylabel("$|u - U|$")
plt.axis((a, b, None, None))
plt.grid(True)

plt.plot(X, np.abs(U - Um), 'r')

plt.show()

