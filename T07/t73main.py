"""Розв'язування лінійної крайової задачі для двовимірного рівняння параболічного типу за допомогою ЛОС (ваги = 0.5)
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg


def t73_f1(x, y, t):
    pass


def t73_f2(x, y, t):
    pass


def t73_mu0x(x, t):
    pass


def t73_mu1x(x, t):
    pass


def t73_mu0y(y, t):
    pass


def t73_mu1y(y, t):
    pass


def t73_u0(x, y):
    pass


# параметри задачі
a1, a2 = 1, 2

# параметри різницевої схеми
Nx = 21
Nx1 = Nx - 1
X = np.linspace(0, a1, Nx)
hx = X[1] - X[0]

Ny = 41
Ny1 = Ny - 1
Y = np.linspace(0, a2, Ny)
hy = Y[1] - Y[0]

tau = 5e-2
M = 100

# обчислення додаткових констант
tau2 = 0.5 * tau

gsx = tau2 / (hx * hx)
gs2x = 2 * gsx
ckx = 1 + gs2x

gsy = tau2 / (hy * hy)
gs2y = 2 * gsy
cky = 1 + gs2y

# формування початкових умов при t=0
U = np.zeros((Nx, Ny))
for i, xi in enumerate(X):
    U[i, :] = t73_u0(xi, Y)

Yy, Xx = np.meshgrid(Y, X)

# побудова графіків розв'язку при t=0
t = 0
plt.figure(1)
ax = plt.axes(projection='3d')
ax.set_title("Y(x1, x2, {})".format(t))
ax.plot_surface(Xx, Yy, U, cmap='viridis')

# формування постійних складових матриць СЛАР
Ax = np.full((3, Nx), [[gsx], [-ckx], [gsx]])
fx = np.zeros(Nx)

Ax[0, 1], Ax[1, 0] = 0, -1
Ax[1, -1], Ax[2, -2] = -1, 0

Ay = np.full((3, Ny), [[gsy], [-cky], [gsy]])
fy = np.zeros(Ny)

Ay[0, 1], Ay[1, 0] = 0, -1
Ay[1, -1], Ay[2, -2] = -1, 0

ckx, cky = 1 - gs2x, 1 - gs2y

UJ = np.zeros((Nx, Ny))

for j in range(M):
    # знаходження наближеного розв'язку Y на шарі t=t(j)-tau/2
    t = (j + 1) * tau
    tt = t - tau2

    UJ[:, :] = U

    # обчислення крайових умов
    U[:, 0] = t73_mu0x(X, tt)
    U[:, -1] = t73_mu1x(X, tt)

    U[0, :] = t73_mu0y(Y, tt)
    U[-1, :] = t73_mu1y(Y, tt)

    for k2 in range(1, Ny1):
        yy = Y[k2]
        # формування СЛАР
        fx[0] = -U[0, k2]
        fx[-1] = -U[-1, k2]

        for k1 in range(1, Nx1):
            z = gsx * (UJ[k1 - 1, k2] + UJ[k1 + 1, k2]) + ckx * UJ[k1, k2]
            fx[k1] = - z - tau * t73_f1(X[k1], yy, tt)

        # розв'язок СЛАР, знаходження U[:, k2]
        U[:, k2] = scipy.linalg.solve_banded((1, 1), Ax, fx, overwrite_b=True, check_finite=False)

    # знаходження наближеного розв'язку Y на шарі t=t(j)
    UJ[:, :] = U

    # обчислення крайових умов
    U[:, 0] = t73_mu0x(X, t)
    U[:, -1] = t73_mu1x(X, t)

    for k2 in range(1, Ny1):
        yy = Y[k2]
        U[0, k2] = t73_mu0y(yy, t)
        U[-1, k2] = t73_mu1y(yy, t)

    for k1 in range(1, Nx1):
        xx = X[k1]
        # формування СЛАР
        fy[0] = - U[k1, 0]
        fy[-1] = - U[k1, -1]

        for k2 in range(1, Ny1):
            z = gsy * (UJ[k1, k2-1] + UJ[k1, k2+1]) + cky * UJ[k1, k2]
            fy[k2] = - z - tau * t73_f2(xx, Y[k2], tt)

        # розв'язок СЛАР, знаходження Y[k1,:]
        U[k1, :] = scipy.linalg.solve_banded((1, 1), Ay, fy, overwrite_b=True, check_finite=False)

    if (j + 1) * 2 == M:  # побудова графіка розв'язку при t=M*tau/2
        plt.figure(2)
        ax = plt.axes(projection='3d')
        ax.set_title("Y(x1, x2, {})".format(t))
        ax.plot_surface(Xx, Yy, U, cmap='viridis')


# побудова графіка розв'язку при M*tau
plt.figure(3)
ax = plt.axes(projection='3d')
ax.set_title("Y(x1, x2, {})".format(t))
ax.plot_surface(Xx, Yy, U, cmap='viridis')


# порівняння точного розв'язку U(x1,x2,t)=t+(2-exp(-t))*(x1**2+x2**2) та  наближеного Y в точці t=M*tau
u = lambda x, y, t: t + (2 - np.exp(-t))*(x*x + y*y)

U_true = u(Xx, Yy, t)

U_eps = np.abs(U_true - U)

plt.figure(4)
ax = plt.axes(projection='3d')
ax.set_title("Абсолютна похибка")
ax.plot_surface(Xx, Yy, U_eps, cmap='viridis')

print("Похибка: ||Y - U||c =", np.max(U_eps))

plt.show()

