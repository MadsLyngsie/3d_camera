import numpy as np
from scipy.optimize import least_squares

# def sphere_mini_LM(x, length_normals, ehh):
#     print(x,'x')
#     print(length_normals,'e\length_normals')
#     print(ehh,'ehh')
#     return (1/length_normals[0])*np.exp2(np.abs(np.abs(x[0]-x[1])+length_normals[1]))
#
# length_normals = np.array([[200,300],[300,200]])
#
# x0 = np.array([2,3])
#
# res_lsq_lm = least_squares(sphere_mini_LM, x0 , args = (length_normals))
#
#
# print(res_lsq_lm,'res_lsq_lm')


a = np.ones([673,3])

b = np.ones([3,1])

print(a-np.transpose(b),'dot')
