import cmath

import numpy as np
import scipy.linalg
from scipy.integrate import solve_ivp, quad
import sympy
import matplotlib.pyplot as plt


def firlin(X, D, fir_a, fir_f):
    """Розв'язування методом "стрiльби" системи 2-х лiнiйних дифрiвнянь 1-го порядку
        U'(x) = A(x)*U(x) + F(x), x є (a,b),
    для якої в точках a == X[0], b == X[-1] виконуються крайові умови
        (D[0] * U(1) + D[1] * U[2])|x=a == D[2],
        (D[3] * U(1) + D[4] * U(2))|x=b == D[5],
    Для параметрiв D[0], D[1] i D[3], D[4] виконуються умови:
        D[0]**2 + D[1]**2 != 0 і D[3]**2 + D[4]**2 != 0).

    Вхідні дані
    -----------
    X : масив значеннь вузлiв виведення результатів.

    D : масив значеннями коефiцiєнтiв крайових умов.

    fir_a : опис матриці A(x).

    fir_f : опис вектор-функції F(x).

    Вихідні дані
    ------------
    U : масив з наближеного розв'язку.
    """
    assert D.size > 5, "Not enough input arguments D"
    assert D[0] != 0 or D[1] != 0, "D[0] == D[1] == 0"
    assert D[3] != 0 or D[4] != 0, "D[0] == D[1] == 0"

    n = X.size

    # Розв'язок однорiдної системи ЗДР
    if D[1] != 0:
        u0 = np.array([1.0, D[2] / D[1]])
    else:
        u0 = np.array([D[2] / D[0], 1.0])

    fir_fo = lambda t, y: np.dot(fir_a(t), y)  # опис правої частини для однорідної системи

    U0 = np.zeros((2, n))
    U0[:, 0] = u0
    for i in range(1, n):
        sol = solve_ivp(fir_fo, (X[i-1], X[i]), u0, method='RK45', vectorized=True)
        u0 = sol.y[:, -1]
        U0[:, i] = u0


    # Розв'язок неоднорiдної системи ЗДР
    if D[1] != 0.0:
        u0 = np.array([0.0, D[2] / D[1]])
    else:
        u0 = np.array([D[2] / D[0], 0.0])

    fir_fno = lambda t, y: np.dot(fir_a(t), y) + fir_f(t)  # опис правої частини для неоднорідної системи

    U1 = np.zeros((2, n))
    U1[:, 0] = u0
    for i in range(1, n):
        sol = solve_ivp(fir_fno, (X[i-1], X[i]), u0, method='RK45', vectorized=True)
        u0 = sol.y[:, -1]
        U1[:, i] = u0


    # обчислення параметра c
    c = D[3] * U0[-1, 0] + D[4] * U0[-1, 1]
    c = (D[5] - D[3] * U1[-1, 0] - D[4] * U1[-1, 1]) / c

    # знаходження розв'язку в точках виведення
    U = U1 + c * U0
    return U


# проективні методи

def mm_coloc(pfun, ffun, b, D, eps):
    """Розв'язок крайової задачі
        u"(x) - p(x) * u(x) = -f(x),
        p(x) > 0, x є (0,b),
        u(0) = D[0], u(b) = D[1],
    наближеним проекційним методом - методом колокації.

    Вхідні дані
    -----------
    pfun : функція користувача, з описом p(x).

    ffun : функція користувача, з описом f(x).

    b : права границя області інтегрування.

    D : вектор крайових умов.

    eps : точність (eps>0).

    Вихідні дані
    ------------
    y : поліном (numpy.poly1d) наближеного розв'язку.
    """
    assert b > 0, "{} > 0".format(b)

    f0 = np.poly1d([(D[1] - D[0]) / b, D[0]])
    Y = f0

    for k in range(2, 16):
        YY = Y
        # обчислюємо рівномірні точки колокації
        N = k + 2
        x = np.linspace(0, b, N)[1:N-1]

        # формуємо СЛАР в точках колокації
        A = np.zeros((k, k))
        B = np.zeros(k)

        for j, xj in enumerate(x):
            px = pfun(xj)
            for i in range(k):
                fi = np.poly1d(np.concatenate(([1, -b], np.zeros(i + 1))))
                ddf = fi.deriv(m=2)
                fi *= -px
                fi += ddf
                A[j, i] = fi(xj)

            B[j] = -ffun(xj) + px * f0(xj)

        # розв'язуємо СЛАР
        c = np.linalg.solve(A, B)

        # підставляємо знайдені константи ci в наближений розв'язок
        Y = f0
        for i, ci in enumerate(c):
            fi = np.poly1d(np.concatenate(([ci, -ci * b], np.zeros(i+1))))
            Y += fi

        # порівнюємо сусідні наближення ||Y-YY|| < eps
        fi = (Y - YY).integ()
        if np.abs(fi(b) - fi(0)) < eps:
            break

    return Y


def sym_coloc(pfun, ffun, a, b, D, eps, verbose=False):
    """Розв'язок крайової задачі
        u"(x) - p(x)*u(x) = -f(x),
        p(x)>0, x є (a,b),
        u(0) = D[0], u(b) = D[1],
    наближеним проекційним методом - методом колокації.

    Вхідні дані
    -----------
    pfun : функція користувача, з описом p(x).

    ffun : функція користувача, з описом f(x).

    a : ліва границя області інтегрування.

    b : права границя області інтегрування.

    D : вектор крайових умов.

    eps : точність (eps>0).

    Вихідні дані
    ------------
    y : поліном (numpy.poly1d) наближеного розв'язку.
    """
    assert a < b, "{} < {}".format(a, b)

    x, A, B = sympy.symbols('x A B')
    # обчислюємо phi(0)
    phi0 = A + B * x
    R = sympy.solve((phi0.subs(x, a) - D[0], phi0.subs(x, b) - D[1]))
    phi0 = phi0.subs(((A, R[A]), (B, R[B])))
    Y = phi0

    C = [sympy.symbols('C' + str(k)) for k in range(1, 16)]

    if verbose:
        X0 = np.linspace(a, b, 51)

    for k in range(2, 16):
        YY = Y
        # обчислюємо рівномірні точки колокації
        N = k + 2
        X = np.linspace(a, b, N)[1:N - 1]

        # формуємо наближення u~Y(n)
        Y = phi0
        for i in range(k):
            Y += C[i] * (x - a)**(i+1) * (b - x)


        # формуємо нев'язку
        E = sympy.diff(Y, x, 2) - pfun(x) * Y + ffun(x)

        # формуємо СЛАР в точках колокації
        Ep = tuple(E.subs(x, xk) for xk in X)

        # розв'язуємо СЛАР
        CC = sympy.solve(Ep, *(C[:k]))

        # підставляємо знайдені константи Ck в наближений розв'язок
        Y = Y.subs(CC)

        # порівнюємо сусідні наближення ||Y-YY|| < eps
        if np.abs(sympy.integrate(Y - YY, (x, a, b))) < eps:
            break

        if verbose:
            # візуалізація сусідніх наближень
            u1 = np.vectorize(sympy.lambdify(x, Y,  "numpy"))
            u2 = np.vectorize(sympy.lambdify(x, YY, "numpy"))

            plt.title("Ітерація: {}".format(k-1))
            plt.xlabel("x")
            plt.ylabel("y")
            plt.axis((a, b, None, None))
            plt.grid(True)

            plt.plot(X0, u1(X0))
            plt.plot(X0, u2(X0), 'r', linewidth=0.9)

            plt.legend(("Поточний розв'язок", "Попередній розв'язок"))

            plt.show()

    return Y


def mm_galer(pfun, ffun, b, D, eps):
    """Розв'язок крайової задачі
        u"(x) - p(x)*u(x) = -f(x),
        p(x)>0, x є (0,b),
        u(0) = D[0], u(b) = D[1].
    наближеним проекційним методом - м.Бубнова-Гальоркіна.

    Вхідні дані
    -----------
    pfun : функція користувача, з описом p(x).

    ffun : функція користувача, з описом f(x).

    b : права границя області інтегрування.

    D : вектор крайових умов.

    eps : точність (eps>0).

    Вихідні дані
    ------------
    y : поліном (numpy.poly1d) наближеного розв'язку.
    """
    assert b > 0, "{} > 0".format(b)

    f0 = np.poly1d([(D[1] - D[0]) / b, D[0]])
    df0 = f0.deriv()
    Y = f0

    for k in range(2, 16):
        YY = Y
        A = np.zeros((k, k))
        B = np.zeros(k)

        for j in range(k):
            fj = np.poly1d(np.concatenate(([1, -b], np.zeros(j + 1))))
            dfj = fj.deriv()
            for i in range(k):
                fi = np.poly1d(np.concatenate(([1, -b], np.zeros(i + 1))))
                dfi = fi.deriv()

                z = (dfj * dfi).integ()
                a1 = z(b) - z(0)

                z = fj * fi
                y1 = lambda t: z(t) * pfun(t)
                a2 = quad(y1, 0, b)[0]

                A[j, i] = a1 + a2

            z = (df0 * dfj).integ()
            a1 = z(b) - z(0)

            y1 = lambda t: (ffun(t) - pfun(t) * f0(t)) * fj(t)
            a2 = quad(y1, 0, b)[0]

            B[j] = a2 - a1

        c = np.linalg.solve(A, B)

        Y = f0
        for i, ci in enumerate(c):
            fi = np.poly1d(np.concatenate(([ci, -ci * b], np.zeros(i + 1))))
            Y += fi

        fi = (Y - YY).integ()
        if np.abs(fi(b) - fi(0)) < eps:
            break

    return Y


# Варіаційний метод

def mm_ritz(pfun, ffun, b, D, eps):
    """Розв'язок крайової задачі
        u"(x) - p(x)*u(x) = -f(x),
        p(x) > 0, x є (0,b),
        u(0)=D[0], u(b)=D[1]
    наближеним варіаційним методом - методом Рітца.

    Вхідні дані
    -----------
    pfun : функція користувача, з описом p(x).

    ffun : функція користувача, з описом f(x).

    b : права границя області інтегрування.

    D : вектор крайових умов.

    eps : точність (eps>0).

    Вихідні дані
    ------------
    y : поліном (numpy.poly1d) наближеного розв'язку.
    """
    assert b > 0, "{} > 0".format(b)

    f0 = np.poly1d([(D[1] - D[0]) / b, D[0]])
    Y = f0

    for k in range(2, 16):
        YY = Y
        A = np.zeros((k, k))
        B = np.zeros(k)

        for j in range(k):
            fj = np.poly1d(np.concatenate(([1, -b], np.zeros(j + 1))))
            for i in range(k):
                fi = np.poly1d(np.concatenate(([1, -b], np.zeros(i + 1))))
                ddfi = fi.deriv(m=2)

                z = (fj * ddfi).integ()
                a1 = z(b) - z(0)

                z = fj * fi
                y1 = lambda t: pfun(t) * z(t)
                a2 = quad(y1, 0, b)[0]

                A[j, i] = a1 - a2

            y1 = lambda t: (pfun(t) * f0(t) - ffun(t)) * fj(t)

            B[j] = quad(y1, 0, b)[0]

        c = np.linalg.solve(A, B)

        Y = f0
        for i, ci in enumerate(c):
            fi = np.poly1d(np.concatenate(([ci, -ci * b], np.zeros(i + 1))))
            Y += fi

        fi = (Y - YY).integ()
        if np.abs(fi(b) - fi(0)) < eps:
            break

    return Y


# різницева схема

def m_grid2(pfun, ffun, a, b, D, n):
    """Наближений розв'язок лiнiйного диференцiального рiвняння
        u"(x) - p(x)*u(x) = -f(x),
        p(x)>0, x є (a,b),
    для якого виконуються крайові умови
        D[0]*u'(a) + D[1]*u(a) = -D[2],
        D[3]*u'(b) + D[4]*u(b) = D[5].
    (різницева схемах 2-го порядку точності).
    Для параметрiв D[0], D[1] i D[3], D[4] виконуються умова:
        D[0]**2 + D[1]**2 != 0, D[3]**2 + D[4]**2 != 0.

    Вхідні дані
    -----------
    pfun : функція користувача, з описом p(x).

    ffun : функція користувача, з описом f(x).

    b : права границя області інтегрування.

    D : вектор крайових умов.

    eps : точність (eps>0).

    Вихідні дані
    ------------
    X : масив зi значеннями вузлiв сiтки.

    Y : масив зi значеннями наближеного розв'язку.
    """
    assert a < b, "{} < {}".format(a, b)
    assert D[0] != 0 or D[1] != 0, "D[0] == D[1] == 0"
    assert D[3] != 0 or D[4] != 0, "D[0] == D[1] == 0"

    # формування вузлiв
    X = np.linspace(a, b, n)
    h = X[1] - X[0]

    # допомiжнi змiннi
    n1 = n - 1
    hh = h * h
    h2 = 2 * h

    # формування коефiцiєнтiв Ak, Ck, Bk, Fk, k=1,...,n для СЛАР
    A = np.ones((3, n))
    f = np.zeros(n)

    A[2, 0] = 1
    A[1, 0] = h2 * D[1] - (2 + hh * pfun(a)) * D[0]
    A[0, 1] = 2 * D[0]
    f[0] = - h2 * D[2] - hh * D[0] * ffun(a)

    for k in range(1, n1):
        xk = X[k]
        A[1, k] = - 2 - hh * pfun(xk)
        f[k] = - hh * ffun(xk)

    A[2, n1-1] = 2 * D[3]
    A[1, n1] = -(2 + hh * pfun(b)) * D[3] - h2 * D[4]
    A[0, n1] = 1
    f[n1] = - hh * D[3] * ffun(b) - h2 * D[5]

    Y = scipy.linalg.solve_banded((1, 1), A, f, overwrite_ab=True, overwrite_b=True, check_finite=False)

    return X, Y


def m_iim(kfun, pfun, ffun, a, b, D, n):
    """Наближений розв'язок рiвняння
        (k(x) * u'(x))' - p(x) * u(x) = -f(x),
        k(x) >= k0 > 0, p(x) >= 0, x є (a,b),
        (k, p, f можуть бути розривні)
        u(a) = D[0], u(b) = D[1].
    Різницева схема побудована інтегро-інтерполяційним методом.

    Вхідні дані
    -----------
    kfun : функцiя користувача, з описом k(x).

    pfun : функцiя користувача, з описом p(x).

    ffun : функцiя користувача, з описом f(x).

    D : масив зi значеннями коефiцiєнтiв крайових умов.

    Вихідні дані
    ------------
    X : масив зi значеннями вузлiв сiтки.

    Y : масив зi значеннями наближеного розв'язку.
    """
    X = np.linspace(a, b, n)
    h = X[1] - X[0]

    n1 = n - 1
    hh = h * h

    # формування коефiцiєнтiв Ak, Ck, Bk, Fk, k=1,...,n для СЛАР
    A = np.zeros((3, n))
    f = np.zeros(n)

    A[1, 0] = -1
    f[0] = -D[0]
    aip = kfun(X[1])

    for k in range(1, n1):
        ai = aip
        aip = kfun(X[k+1])

        A[2, k-1] = ai
        A[0, k+1] = aip
        xk = X[k]
        A[1, k] = - ai - aip - hh * pfun(xk)

        f[k] = - hh * ffun(xk)

    A[1, n1] = -1
    f[n1] = -D[1]

    Y = scipy.linalg.solve_banded((1, 1), A, f, overwrite_ab=True, overwrite_b=True, check_finite=False)

    return X, Y


