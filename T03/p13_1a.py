import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# вхiднi данi
a = 0
b = 5

Y0 = np.array([1])  # вектор початкових умов
f31a = lambda t, u: (np.exp(-t) - u[0])  # підфункція правої частини системи

# наближений розв'язкок, отриманий методом Рунге-Кутта 2-го порядку
sol1 = solve_ivp(f31a, (a, b), Y0, method='RK23')
T1 = sol1.t
Y1 = sol1.y[0]

# наближений розв'язкок, отриманий методом Рунге-Кутта 4-го порядку
sol2 = solve_ivp(f31a, (a, b), Y0, method='RK45')
T2 = sol2.t
Y2 = sol2.y[0]

X = np.linspace(0, 5, 51)

# точний розв'язок
u = lambda x: np.exp(-x) * (1 + x)

plt.title("$u^{\prime}(x) = e^{-x} - u(x)$")
plt.xlabel('x')
plt.ylabel('y')
plt.plot(T1, Y1, T2, Y2, X, u(X))

plt.legend(("RK23", "RK45", "точний"))

plt.show()

