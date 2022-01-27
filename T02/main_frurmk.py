import matplotlib.pyplot as plt
import numpy as np

from integral_equation import frur_mk

# вхiднi данi
a = 0
b = 0.35

# параметри сітки і ітераційного процесу
n = 13
eps = 1e-8
mi = 500

# початкове наближення
Y0 = np.full(n, 0.05)

def frurmk_k(x,t,u):
    # опис ядра K(x,t,u)
    y = np.exp(x - u)
    # опис похідної dK(x,t,u)/du
    yp = -y
    return y, yp


def frurmk_f(x,u):
    # опис правої частини f(x,u)
    y = x - u
    # опис похідної df(x,u)/du
    yp = -1
    return y, yp


X, Y = frur_mk(a, b, 1, n, frurmk_k, frurmk_f, Y0, eps, mi, 2)

plt.figure(3)
plt.title("Наближений розв'язок IP")
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

plt.plot(X, Y, 'm')

plt.show()

