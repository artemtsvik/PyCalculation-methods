import matplotlib.pyplot as plt
import numpy as np

from ode import mm_coloc


a, b = 0, 1
D = np.zeros(2)

pfun = lambda x: 0
ffun = lambda x: 6 * x * (2 * x - 1)
u = lambda x: (1 - x) * x**3

eps = 1e-5

Y = mm_coloc(pfun, ffun, b, D, eps)

print(Y)

nx = 51
x = np.linspace(a, b, nx)
y = Y(x)

U = u(x)

plt.figure(1)
plt.title("Метод колокації")
plt.xlabel("x")
plt.ylabel("u")
plt.axis((a, b, None, None))

plt.plot(x, y)

plt.figure(2)
plt.title("Абсолютна похибка")
plt.xlabel("x")
plt.ylabel("$|u - u_{true}|$")
plt.axis((a, b, None, None))
plt.grid(True)

plt.plot(x, np.abs(y - U), 'r', linewidth=0.8)

plt.figure(3)
plt.title("Відносна похибка")
plt.xlabel("x")
plt.ylabel("$\left|\\frac{u}{u_{true}} - 1 \\right|$")
plt.axis((a, b, None, None))
plt.grid(True)

plt.plot(x, np.abs(y / U - 1), 'm', linewidth=0.8)


plt.show()

