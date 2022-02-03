"""Розв'язання гіперболічного рівняння першого порядку з постійним коефіцієнтом (неявна схема)"""
import matplotlib.pyplot as plt
import numpy as np


# Завдання параметрів задачі
a, b, c = 0, 3, 1
f = lambda x, t: 3 + x*(1 - x) + t * (3 - 2*x)
u0 = lambda x: x * (3 - x)
g1 = lambda t: 0

# Завдання параметрів сіток
N = 301
X = np.linspace(a, b, N)

h = X[1] - X[0]
h2 = 0.5 * h
tau = 0.01
M = 500
tau2 = 0.5 * tau

# Обчислення додаткових констант
C = c * tau / h
B = 1 / (1 + C)
A = C * B
C = tau * B

# Обчислення початкових умов
t = 0
y = u0(X)
U0 = y.copy()

# Визначення розв'язку на шарах сітки
for j in range(M):
    t += tau
    y[0] = g1(t)

    for k in range(1, N):
        y[k] = A * y[k-1] + B * y[k] + C * f(X[k] - h2, t - tau2)

    if (j + 1) * 2 == M:
        U1 = y.copy()


# побудова графіків розв'язку при t=0, M*tau/2, M*tau
T1 = M * tau
T = t / 2

plt.figure(1)
plt.title("Наближений розв'язок рівняння переносу")
plt.xlabel("x")
plt.ylabel("u")
plt.axis((a, b, None, None))

plt.plot(X, U0, 'b.-', markevery=10)
plt.plot(X, U1, 'k.--', markevery=10)
plt.plot(X, y, 'r*-', markevery=10)

plt.legend(("u(x,0)", "u(x,{})".format(T), "u(x,{})".format(T1)))


u = lambda x, t: (t + 1) * x * (3 - x)
U = u(X, t)

plt.figure(2)
plt.title("Абсолютна похибка")
plt.xlabel("x")
plt.ylabel("$|u - U|$")
plt.axis((a, b, None, None))
plt.grid(True)

plt.plot(X, np.abs(y - U), 'r')

plt.show()

