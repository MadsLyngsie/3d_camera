import numpy as np
from sympy import *
from sympy.vector import *
from numpy import linalg as LA
from sympy.interactive.printing import init_printing
init_printing(use_unicode=False, wrap_line=False)
from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt


a = np.asarray([1,2,3])

b = np.asarray([[1,2,3],[1,2,3],[1,2,3],[1,2,3]])
print(a)
print(a-b)

r,x,y,z = symbols('r,x,y,z')

c = Matrix([ x**2 , y**3 ,  z**4])

mi = np.array([[1,2,3],[1,2,3],[1,2,3],[1,2,3]])

# mi = np.array([1,2,3])

c = Matrix([1,2,3])

m = Matrix(mi)

print(m,'m')
print(c,'c')


print(c.T-m)

# fx = (mi - c).dot(mi - c) - r
#
# print(diff(fx, x))
#
#
# print(np.dot(np.array([1,1,1]),np.array([1,1,1])))
