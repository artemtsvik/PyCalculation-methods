"""Розв'язування квазілінійної крайової задачі для одновимірного рівняння типу Шредінгера
різницевим методом (симетрична схема)
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg


# параметри задачі
D, l = 5e-3j, 6.
l2 = l / 2


def t10_u0(x):
    if 0. < x < l:
        return (1 + 1j) * np.exp(-40. * (x - l2)**2)
    else:
        return 0


t10_u0 = np.vectorize(t10_u0, otypes=[complex])


def t10_f(u):
    return (0.1j * np.abs(u)**2) / (1 + np.abs(u)**2)


# параметри різницевої схеми
sigma = 0.5
N = 301
N1 = N - 1
X = np.linspace(0, l, N)
h = X[1] - X[0]

tau = 2e-2
M = 500
zz = tau * M
Z = np.linspace(0, zz, M + 1)
Intu = Z.copy()

eps = 1e-8
mk = 1000

# обчислення додаткових констант (з урахуванням крайових умов)
gam = D * tau / (h * h)
gs = gam * sigma
gs2 = 2 * gs
ck = 1 + gs2
s1 = 1 - sigma
gs1 = gam * s1
ckj = 1 - 2 * gs1
# формування початкових умов при z=0 і матриці СЛАР
A = np.full((3, N), [[gs], [-ck], [gs]], dtype="complex")

A[0, 1] = 0
A[1, 0] = A[1, -1] = -1
A[2, -2] = 0

f = np.zeros(N, dtype="complex")
fj = f.copy()

U = t10_u0(X)
U0 = U.copy()  # запам'ятовуєм для побудови графіку розв'язку в z=0

ck = h * np.sum(np.abs(U)**2)
Intu[0] = ck  # закон збереження

Uj = np.zeros(N, dtype="complex")
Us = Uj.copy()

for j in range(M):
    # знаходження наближеного розв'язку Y(:) на черговому шарі z=z(j)
    zj = Z[j+1]
    Uj[:] = U

    # формуєм постійну частину СЛАР
    for k in range(1, N1):
        fj[k] = gs1 * (U[k-1] + U[k+1]) + ckj * U[k]

    for s in range(mk):
        Us[:] = U

        # формування правої частини СЛАР
        f[0] = f[-1] = 0
        for k in range(1, N1):
            gs = sigma * U[k] + s1 * Uj[k]
            f[k] = - fj[k] - tau * t10_f(gs) * gs

        # розв'язок СЛАР, знаходження U
        U[:] = scipy.linalg.solve_banded((1, 1), A, f, overwrite_b=True, check_finite=False)

        if np.all(np.abs(U - Us) < eps):
            break

    if (j + 1) * 2 == M:
        U1 = U.copy()  # запам'ятовуєм для побудови графіку в t=M*tau/2

    gs = h * np.sum(np.abs(U)**2)
    Intu[j+1] = gs

    if s + 1 >= mk:
        break  # у випадку не збіжності МПІ


# побудова графіків розв'язку при z=0, Z/2, Z
T1 = zz / 2

plt.figure(1)
plt.xlabel("$x$")
plt.ylabel("$|u|$")
plt.axis((0, l, None, None))

plt.plot(X, np.abs(U0), 'b.-', markevery=6)
plt.plot(X, np.abs(U1), 'k.--', markevery=6)
plt.plot(X, np.abs(U), 'r*-', markevery=6)

plt.legend(('$|u(x, 0)|$', '$|u(x, {})|$'.format(T1), '$|u(x, {})|$'.format(zz)))


plt.figure(2)
plt.xlabel("$x$")
plt.ylabel("arg(u)")
plt.axis((0, l, None, None))

plt.plot(X, np.angle(U0), 'b.-', markevery=6)
plt.plot(X, np.angle(U1), 'k.--', markevery=6)
plt.plot(X, np.angle(U), 'r*-', markevery=6)

plt.legend(("arg(u(x, 0))", "arg(u(x, {}))".format(T1), "arg(u(x, {}))".format(zz)))

plt.figure(3)
plt.xlabel("$z$")
plt.ylabel("$||u(x, z)||^{2} - ||u_{0}(x)||^{2}$")
plt.axis((0, zz, None, None))
plt.grid(True)

plt.plot(Z, Intu - Intu[0], 'r')

print(np.linalg.norm(Intu - Intu[0], np.inf))

plt.show()

