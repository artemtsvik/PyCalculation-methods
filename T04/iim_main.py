import matplotlib.pyplot as plt
import numpy as np

from ode import m_iim


# п.1 Вхідні дані неперервні:
# Розв'язання модельної задачі
#   (k(x) * U'(x))' - p(x) * U(x) = -f(x), x є (0,1),
#   U(0) = 0, U(1) = sin(1).
# Для k(x) = 1 + x, p(x) = exp(-x), f(x) = sin(x)*(1 + x + exp(-x)) - cos(x)
# розв'язок буде U(x) = sin(x).

a, b = 0, 1
D = np.array([0, np.sin(1)])

n = 51

h = (b - a) / (n - 1)
h05 = 0.5 * h

iim_k1 = lambda x: 1 + x - h05
iim_p1 = lambda x: np.exp(-x)
iim_f1 = lambda x: np.sin(x) * (1 + x + np.exp(-x)) - np.cos(x)
iim_u1 = np.sin

X, Y = m_iim(iim_k1, iim_p1, iim_f1, a, b, D, n)

plt.figure(1)
plt.title("РС ІІМ п.1 (k,p,f-неперервні)")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.axis((a, b, None, None))
plt.plot(X, Y)

U = iim_u1(X)  # Точний розв'язок

print("Похибка в нормі C=", np.linalg.norm(Y - U))

plt.figure(2)
plt.title("Абсолютна похибка")
plt.xlabel("x")
plt.ylabel("$\left|y - u(x)\\right|$")
plt.axis((a, b, None, None))
plt.grid(True)

plt.plot(X, np.abs(U - Y), 'r')

plt.show()


# п.2 Вхідні дані розривні f(x):
# Розв'язання модельної задачі
#   U"(x) = f(x), x є (-1,1)
#   U(-1) = 0, U(1) = 1.
# Для k(x) = 1, p(x) = 0, f(x) = { 100*sign(x) + exp(x) & x#0 | 0 & x=0 }
# розв'язок буде
# U(x)=0.5*(1 - 99*x) + 50*x**2*sign(x) + exp(x) - 0.5*((1+x)*exp(1) + (1 - x)*exp(-1)).

a, b = -1, 1
D = np.array([0, 1])

n = 51
h = (b - a)/(n - 1)
h05 = 0.5 * h

iim_k2 = lambda x: 1
iim_p2 = lambda x: 0
iim_f2 = lambda x: -100 * np.sign(x) - np.exp(x) if x != 0 else -0.5 * (np.exp(-h05) + np.exp(h05))
iim_u2 = lambda x: (0.5 * (1 - 99 * x) + 50 * x * x * np.sign(x) +
                    np.exp(x) - 0.5 * ((1 + x) * np.exp(1) + (1 - x)*np.exp(-1)))

X, Y = m_iim(iim_k2, iim_p2, iim_f2, a, b, D, n)

plt.figure(1)
plt.title("РС ІІМ п.2 (f-розривна)")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.axis((a, b, None, None))
plt.plot(X, Y)

U = iim_u2(X)  # Точний розв'язок

print("Похибка в нормі C=", np.linalg.norm(Y - U))

plt.figure(2)
plt.title("Абсолютна похибка")
plt.xlabel("x")
plt.ylabel("$\left|y - u(x)\\right|$")
plt.axis((a, b, None, None))
plt.grid(True)

plt.plot(X, np.abs(U - Y), 'r')

plt.show()


# п.3 Вхідні дані розривні k(x):
# Розв'язання модельної задачі
#   (k(x) * U'(x))' = 0, x є (0,1),
#   U(0) = 1, U(1) = 0.
# Для k(x) = { k1 & 0<=x<=t | k2 & t<=x<=1 },
#   t = 38.5 * h, 1<m<n; p(x) = 0, f(x) = 0
# розв'язок буде
# U(x) = { 1-al*x & 0<=x<=t | be*(1 - x) & t<=x<=1 },
# де g = k1/k2, al = 1/(g + (1-g)*t), be = al*g.

a, b = 0, 1
D = np.array([1, 0])
k1, k2 = 2, 3.5

n = 101
h = (b - a)/(n - 1)
h05 = 0.5 * h
t = a + 38.5 * h

g = k1 / k2
a1 = 1 / (g + (1 - g) * t)
be = a1 * g


def iim_k3(x):
    if x < t:
        return k1
    elif x == t + h05:
        return 0.5 * (k1 + k2)
    else:
        return k2


iim_f3 = lambda x: 0
iim_u3 = lambda x: (1 - a1 * x) if x <= t else be * (1 - x)
iim_u3 = np.vectorize(iim_u3)

X, Y = m_iim(iim_k3, iim_p2, iim_f3, a, b, D, n)

plt.figure(1)
plt.title("РС ІІМ п.3 (k-розривна)")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.axis((a, b, None, None))
plt.plot(X, Y)

U = iim_u3(X)  # Точний розв'язок

print("Похибка в нормі C=", np.linalg.norm(Y - U))

plt.figure(2)
plt.title("Абсолютна похибка")
plt.xlabel("x")
plt.ylabel("$\left|y - u(x)\\right|$")
plt.axis((a, b, None, None))
plt.grid(True)

plt.plot(X, np.abs(U - Y), 'r')

plt.show()


# п.4 Вхідні дані розривні p(x):
# Розв'язання задачі
#   U"(x) - p(x)*U(x) = 0, x є (0,1),
#   U(0) = 1, U(1) = 0.
# Для p(x) = { p1 & 0<=x<t | p2 & t<x<=1 }, t = X(m), 1<m<n; k(x) = 1, f(x) = 0.

a, b = 0, 1
D = np.array([1, 0])
p1, p2 = 2, 10

n = 101
h = (b - a)/(n - 1)
h05 = 0.5 * h
t = a + 38.5 * h


def iim_p4(x):
    if x < t:
        return p1
    elif x == t + h05:
        return 0.5 * (p1 + p2)
    else:
        return p2


X, Y = m_iim(iim_k2, iim_p4, iim_f3, a, b, D, n)

plt.figure(1)
plt.title("РС ІІМ п.4 (p-розривна)")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.axis((a, b, None, None))
plt.grid(True)

plt.plot(X, Y)

plt.show()

