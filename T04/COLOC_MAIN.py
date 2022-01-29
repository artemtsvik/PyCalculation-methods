import numpy as np
import sympy

from ode import sym_coloc


x = sympy.symbols('x')

a1, b1 = 0, 1
D1 = np.zeros(2)

pfun1 = lambda t: 0
ffun1 = lambda t: 6 * t * (2*t - 1)

eps = 1e-5

Y1 = sym_coloc(pfun1, ffun1, a1, b1, D1, eps, verbose=True)

# точний розв'язок x**3 * (1 - x)
print(Y1)

g1 = sympy.plotting.plot(Y1, (x, a1, b1), title=Y1, show=False)
g1.show()

a2, b2 = 1, 2
D2 = np.array([1, 2])

pfun2 = lambda t: 1 + t
ffun2 = lambda t: 2 * t * (t - 1)

Y2 = sym_coloc(pfun2, ffun2, a2, b2, D2, eps, verbose=True)

print(Y2)

g2 = sympy.plotting.plot(Y2, (x, a2, b2), title=Y2, show=False)
g2.show()

