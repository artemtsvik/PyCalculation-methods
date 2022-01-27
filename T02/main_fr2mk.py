import matplotlib.pyplot as plt

from integral_equation import fr2_mk


# вхідні дані
lambd = 12
a = 0
b = 1
n = 101

fr2mk_k = lambda x, s: 1 / (x + 1)  # опис ядра K(x,s)
fr2mk_f = lambda x: x * x * (1 - x) - 1 / (x + 1)   # опис правої частини f(x)

X, Y = fr2_mk(lambd, a, b, 1, n, fr2mk_k, fr2mk_f)

plt.figure(1)
plt.title("Наближений розв'язок IP")
plt.xlabel('x')
plt.ylabel('y')
plt.plot(X, Y, 'm')


# Рiзниця мiж точним u(x)=x * x * (1-x) та наближеним розв'язком
u = lambda x: x * x * (1 - x)
Z = u(X) - Y

plt.figure(2)
plt.title("Похибка розв'язку u(x) - y(x)")
plt.xlabel('x')
plt.ylabel('u - y')
plt.grid(True)
plt.plot(X, Z, 'r', linewidth=0.8)

plt.show()


