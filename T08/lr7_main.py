"""Лабораторна робота No7 з МО
Розв'язування змішаної крайової задачі для рівняння Пуассона в області складеної з класичних геометричних елементів
(прямокутників, трапецій, трикутників)
"""
import matplotlib.pyplot as plt
import numpy as np


def lr7_f(x, y):
    """Функція f(x,y)"""
    return 2 * (x * (1 - x) + y * (1 - y))


def lr7_mu11x(x):
    """Крайові умови на (0<=x<=0.3) & (y=0.3)"""
    return 0.21 * x * (1 - x)


def lr7_mu12x(x):
    """Крайові умови на (0.3<=x<=1) & (y=0)"""
    return 0


def lr7_mu21x(x):
    """Крайові умови на (0<=x<=0.5) & (y=1)"""
    return 0


def lr7_muxy(x):
    """Крайові умови на (0.5<=x<=1) & (y=2-2x)"""
    xx = 2 * x
    return xx * (1 - x)**2 * (xx - 1)


def lr7_mu11y(y):
    """Крайові умови на (0<y<0.3) & (x=0.3)"""
    return 0.4 * y * (1 - y)


def lr7_mu12y(y):
    """Крайові умови на (0.3<y<1) & (x=0)"""
    return y * (1 - y)


# параметри задачі
a, b = 1, 1
Hx, Hy = 0.005, 0.01
Nx, Ny = 201, 101

X = np.linspace(0, a, Nx)
Y = np.linspace(0, b, Ny)

Nx11, Nx21, Ny11 = 61, 101, 31

rho = 1.965
eps = 1e-8
kmax = 20000

Nx1, Ny1 = Nx - 1, Ny - 1
HxH = 1 / (Hx * Hx)
HyH = 1 / (Hy * Hy)

C = rho * 0.5 / (HxH + HyH)
A = HxH * C
B = HyH * C
D = rho - 1

# формування крайових умов та початкового наближення
U = np.zeros((Ny, Nx))

# (0 <= X < 0.3) & (0.3 = Y)
for i in range(Nx11):
    U[Ny11 - 1, i] = lr7_mu11x(X[i])

# (0.3<= X <= 1) & (0 = Y)
for i in range(Nx11-1, Nx):
    U[0, i] = lr7_mu12x(X[i])

# (0 <= X < 0.5) & (1 = Y)
for i in range(Nx21):
    U[-1, i] = lr7_mu21x(X[i])

# (0.5 <= X <= a) & (2-2*X = Y)
for j in range(1, Ny1):
    i = Nx1 - j
    U[j, i] = lr7_muxy(X[i])

UU = U.copy()

for k in range(kmax):
    UU[:, :] = U
    for j in range(1, Ny11-1):
        i = Nx11 - 1
        u = U[j, i]
        U[j, i] = 2.0*A*(U[j, i+1] - Hx*lr7_mu11y(Y[j])) + B*(U[j-1, i] + U[j+1, i]) - D*u + C*lr7_f(X[i], Y[j])

        for i in range(Nx11, Nx1-j):
            u = U[j, i]
            U[j, i] = A*(U[j, i-1] + U[j, i+1]) + B*(U[j-1, i] + U[j+1, i]) - D*u + C*lr7_f(X[i], Y[j])

    j = Ny11 - 1
    for i in range(Nx11, Nx1-j):
        u = U[j, i]
        U[j, i] = A*(U[j, i-1] + U[j, i+1]) + B*(U[j-1, i] + U[j+1, i]) - D*u + C*lr7_f(X[i], Y[j])

    for j in range(Ny11, Ny1):
        i = 0
        u = U[j, i]
        U[j, i] = 2.0*A*(U[j, i+1] - Hx*lr7_mu12y(Y[j])) + B*(U[j-1, i] + U[j+1, i]) - D*u + C*lr7_f(X[i], Y[j])

        for i in range(1, Nx1 - j):
            u = U[j, i]
            U[j, i] = A*(U[j, i-1] + U[j, i+1]) + B*(U[j-1, i] + U[j+1, i]) - D*u + C*lr7_f(X[i], Y[j])

    if np.all(np.abs(UU - U) < eps):
        break


# побудова графіку розв'язку
print("Кількість ітерацій:", k+1)

Xx, Yy = np.meshgrid(X, Y)

plt.figure(1)
ax = plt.axes(projection='3d')
ax.set_title("Різницевий розв’язок")
ax.plot_surface(Xx, Yy, U, cmap='inferno')

u = lambda x, y: x * (1 - x) * y * (1 - y)

UU[:, :] = np.zeros((Ny, Nx))
for j in range(Ny):
    for i in range(Nx):
        if (j >= Ny11-1 or i >= Nx11-1) and i < Nx - j:
            UU[j, i] = u(X[i], Y[j])

plt.figure(2)
ax = plt.axes(projection='3d')
ax.set_title("Точний (модельний) розв’язок")
ax.plot_surface(Xx, Yy, UU, cmap='viridis')

U_eps = np.abs(U - UU)

plt.figure(3)
ax = plt.axes(projection='3d')
ax.set_title("Абсолютна похибка")
ax.plot_surface(Xx, Yy, U_eps, cmap='hot')

print("Похибка: ||Y - U||c =", np.max(U_eps))

plt.show()

