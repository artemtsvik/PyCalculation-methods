import numpy as np
import scipy.linalg

__all__ = ["mznopj", "ptmopch", "optchset"]


def mznopj(L, N, eps, ffun, mux, muy):
    """Розв'язування задачі Діріхле в прямокутнику
        П U dП = (0 <= x <= L[0] & 0 <= y <= L[1])
    для рівняння Пуассона за допомогою різницевої схеми з використанням методу змінних напрямків і
    вибором оптимальних параметрів по Жордану.
        U"xx + U"yy = -f(x,y), (x,y) є П,
        U(x,y) = mu(x,y), (x,y) є dП.

    Вхідні дані
    -----------
    L : вектор опису прямокутника.

    N : вектор опису кількості точок сітки.

    eps : точність (0<eps<1) ітераційного процесу.

    ffun : опис функції f(x,y).

    mux : опис функцій mu(x,y), для границь y=0, L[1].

    muy : опис функцій mu(x,y), для границь x=0, L[0].

    Вихідні дані
    ------------
    n : кількість кроків ітераційного процесу.

    X : сітка по x.

    Y : сітка по y.

    U : сіткова функція наближеного розв'язку U(x,y).
    """
    assert 0 < eps < 1, "0 < {} < 1".format(eps)

    # параметри різницевої схеми
    Nx = N[0]
    Nx1 = Nx - 1
    X = np.linspace(0, L[0], Nx)
    hx = X[1] - X[0]

    Ny = N[1]
    Ny1 = Ny - 1
    Y = np.linspace(0, L[1], Ny)
    hy = Y[1] - Y[0]

    # формування граничних умов і початкового наближення
    U = np.zeros((Nx, Ny))
    U[0, :], U[-1, :] = muy(Y)

    for k1 in range(1, Nx1):
        z = mux(X[k1])
        U[k1, 0], U[k1, -1] = z

    # обчислення оптимальних ітераційних параметрів
    hhx, hhy = 1 / (hx * hx), 1 / (hy * hy)
    p, ka = 4 * hhx, 4 * hhy
    t = 0.5 * np.pi
    r = t * hx / L[0]
    t *= hy / L[1]

    d1 = np.sin(r)
    d1 *= p * d1
    d2 = np.sin(t)
    d2 *= ka * d2

    D1 = np.cos(r)
    D1 *= p * D1
    D2 = np.cos(t)
    D2 *= ka * D2

    ka = (D1 - d1) / (D2 + d1)
    t = ka * (D2 - d2)
    t = np.sqrt(t / (D1 + d2))

    eta = (1 - t) / (1 + t)

    ka *= D2 / D1
    p = (ka - t) / (ka + t)

    r = D1 - D2 + (D1 + D2) * p
    r *= 0.5 / (D1 * D2)
    q = r + (1 - p) / D1

    n = int(np.log(4 / eps) * np.log(4 / eta) / (np.pi * np.pi))
    tet = eta * eta
    tet *= 0.0625 * (1 + 0.5 * tet)
    ka = 2 * n
    eta = 0.5 + tet

    # формування складових матриць СЛАР
    Ax = np.full((3, Nx), [[0.], [-1.], [0.]])
    fx = np.zeros(Nx)

    Ay = np.full((3, Ny), [[0.], [-1.], [0.]])
    fy = np.zeros(Ny)

    UJ = np.zeros((Nx, Ny))

    for j in range(n):
        # знаходження компоненти w ітераційних параметрів
        t = (2 * j + 1) / ka
        d1 = eta * (1 + tet**t)
        d2 = 1 + tet**(1 - t) + tet**(1 + t)
        d2 *= tet**(0.5*t)
        w = d1 / d2

        # знаходження наближеного розв'язку Y для (j+1/2)
        tau = (q * w + r) / (1 + w * p)
        UJ[:, :] = U

        d1 = tau * hhx
        d2 = 1 + 2 * d1
        D1 = tau * hhy
        D2 = 1 - 2 * D1

        for k1 in range(1, Nx1):
            Ax[2, k1 - 1] = d1
            Ax[1, k1] = -d2
            Ax[0, k1 + 1] = d1

        for k2 in range(1, Ny1):
            fx[0] = - U[0, k2]
            fx[-1] = - U[-1, k2]

            yy = Y[k2]
            for k1 in range(1, Nx1):
                z = D1 * (UJ[k1, k2 - 1] + UJ[k1, k2 + 1]) + D2 * UJ[k1, k2]
                fx[k1] = - z - tau * ffun(X[k1], yy)

            U[:, k2] = scipy.linalg.solve_banded((1, 1), Ax, fx, overwrite_b=True, check_finite=False)

        # знаходження наближеного розв'язку Y для (j+1)
        tau = (q * w - r)/(1 - w * p)
        UJ[:, :] = U

        d1 = tau * hhy
        d2 = 1 + 2 * d1

        D1 = tau * hhx
        D2 = 1 - 2 * D1

        for k2 in range(1, Ny1):
            Ay[2, k2 - 1] = d1
            Ay[1, k2] = - d2
            Ay[0, k2 + 1] = d1

        for k1 in range(1, Nx1):
            fy[0] = - U[k1, 0]
            fy[-1] = - U[k1, -1]

            xx = X[k1]
            for k2 in range(1, Ny1):
                z = D1 * (UJ[k1 - 1, k2] + UJ[k1 + 1, k2]) + D2 * UJ[k1, k2]
                fy[k2] = - z - tau * ffun(xx, Y[k2])

            U[k1, :] = scipy.linalg.solve_banded((1, 1), Ay, fy, overwrite_b=True, check_finite=False)

    return n, X, Y, U


def ptmopch(L, N, eps, ffun, mux, muy):
    """Розв'язування задачі Діріхле в прямокутнику
        П U dП = (0 <= x <= L[0] & 0 <= y <= L[1])
    для рівняння Пуассона за допомогою різницевої схеми з використанням позмінно-трикутного методу
    з чебишовським набором параметрів (ЧНП).
        U"xx + U"yy = -f(x,y), (x,y) є П,
        U(x,y) = mu(x,y), (x,y) є dП.

    Вхідні дані
    -----------
    L : вектор опису прямокутника.

    N : вектор опису кількості точок сітки.

    eps : точність (0<eps<1) ітераційного процесу.

    ffun : опис функції f(x,y).

    mux : опис функцій мю(x,y), для границь y=0, L[1].

    muy : опис функцій мю(x,y), для границь x=0, L[0].

    Вихідні дані
    ------------
    n : кількість кроків ітераційного процесу.

    X : сітка по x.

    Y : сітка по y.

    U : сіткова функція наближеного розв'язку U(x,y).
    """
    assert 0 < eps < 1, "0 < {} < 1".format(eps)

    # параметри різницевої схеми
    Nx = N[0]
    Nx1 = Nx - 1
    X = np.linspace(0, L[0], Nx)
    hx = X[1] - X[0]

    Ny = N[1]
    Ny1 = Ny - 1
    Y = np.linspace(0, L[1], Ny)
    hy = Y[1] - Y[0]

    # формування граничних умов і початкового наближення
    U = np.zeros((Nx, Ny))
    UJ = U.copy()

    U[0, :], U[-1, :] = muy(Y)
    for k1 in range(1, Nx1):
        z = mux(X[k1])
        U[k1, 0], U[k1, -1] = z

    # обчислення оптимальних ітераційних параметрів
    hhx, hhy = 1 / (hx * hx), 1 / (hy * hy)
    p, ka = 4 * hhx, 4 * hhy
    t = 0.5 * np.pi

    r = t * hx / L[0]
    t *= hy / L[1]

    d = np.sin(r)
    d *= p * d
    D = np.sin(t)
    D *= ka * D

    d += D
    D = p + ka
    ka = d / D
    d2 = np.sqrt(ka)
    d1 = 0.5 * d / (1 + d2)
    d2 = 0.25 * d / d2

    w = 2 / np.sqrt(d * D)
    t = d1 / d2
    r = (1 - t) / (1 + t)
    q = 2 / (d1 + d2)

    n = int(0.5 * np.log(2/eps) / np.sqrt(2 * np.sqrt(ka)))

    M = optchset(n)  # стійкий набір множини ЧНП

    d1, d2 = w * hhx, w * hhy
    d = 1 / (1 + d1 + d2)
    D = -2 * (hhx + hhy)

    for j in range(n):
        tau = q / (1 + r * M[j])  # ітераційний параметр
        for k2 in range(1, Ny1):
            yy = Y[k2]
            for k1 in range(1, Nx1):
                z = hhx * (U[k1-1, k2] + U[k1+1, k2])
                p = hhy * (U[k1, k2-1] + U[k1, k2+1])
                z += p + D * U[k1, k2] + ffun(X[k1], yy)
                p = d1 * UJ[k1-1, k2] + d2 * UJ[k1, k2-1]
                UJ[k1, k2] = (p + z) * d

        for k1 in range(Nx1-1, 0, -1):
            for k2 in range(Ny1-1, 0, -1):
                z = d1 * UJ[k1+1, k2] + d2 * UJ[k1, k2+1]
                UJ[k1, k2] = (z + UJ[k1, k2]) * d

        for k1 in range(1, Nx1):
            U[k1, :] += tau * UJ[k1, :]

    return n, X, Y, U


def optchset(n):
    """Побудова множини оптимальних Чебишовських параметрiв

    Вхiднi данi
    -----------
    n : кiлькiсть чисел множини.

    Вихiднi данi
    ------------
    T : оптимальний набiр параметрiв.
    """
    T = _nchset(n)
    b = np.pi / (2 * n)
    T = - np.cos(b * T)
    return T


def _nchset(n):
    """Побудова множини T(i) непарних чисел для оптимальних Чебишовських параметрiв.

    Вхідні дані
    -----------
    n : кiлькiсть чисел.

    Вихiднi данi
    ------------
    T : числова множина.
    """
    P = [n]
    r = n

    while r != 1:
        if r % 2 == 0:
            r //= 2
        else:
            r -= 1
        P.append(r)

    k = len(P)

    T = np.zeros(n, dtype=np.int32)
    T[0] = 1
    kT = 1
    for j in range(k - 2, -1, -1):
        t = r
        r = P[j]
        if 2 * t == r:
            m = 2 * r
            if j != 0 and P[j - 1] == r + 1:
                m += 2

            for i in range(t-1, -1, -1):
                i2 = 2 * i + 1
                e = T[i]
                T[i2 - 1] = e
                T[i2] = m - e

            kT *= 2
        else:
            kT += 1
            T[kT - 1] = r

    return T

