import numpy as np
from sympy import *
from sympy.vector import *
from numpy import linalg as LA
import pyrealsense2 as rs
import cv2
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from numpy import linalg as LA
from multiprocessing import Pool
from functools import partial
from mpl_toolkits import mplot3d
from numpy.linalg import norm
import imutils
import sys
from camera import Camera
from scipy.optimize import least_squares
import math
from scipy import spatial
from scipy.optimize import curve_fit
import sympy
from sympy.vector import *
import functools
from multiprocessing.pool import ThreadPool
from sympy.utilities.lambdify import lambdify, implemented_function
# from sympy.abc import x, y, z
from sympy import Eq
from numpy.linalg import inv
from sympy.vector import CoordSys3D


class some:
    def __init__(self):
        self.ThreadPool = ThreadPool(processes = 100)
        self.len = 0
        self.a = 0

    def testtesting(self,f,g):

        x,y = symbols('x,y')

        self.a = self.a.subs([(x,f), (y,g)])

        return self.a

    def startingthing(self):

        x,y = symbols('x,y')

        self.a = x + 1 + y

        self.a = s.testtesting(3,3)

        return self.a

    def compute_jacobian_sphere(self, input_vector):

        self.len = len(input_vector)

        mix = input_vector[0]
        miy = input_vector[1]
        miz = input_vector[2]
        cx = input_vector[3]
        cy = input_vector[4]
        cz = input_vector[5]
        r = input_vector[6]


        jacobian = self.jacobian_sphere(mix,miy,miz,cx,cy,cz,r)

        return jacobian

    def jacobian_sphere(self,mix,miy,miz,cx,cy,cz,r):

        cxs = symbols('cxs')
        fcx = (sqrt((mix - cxs)**2 - (miy - cy)**2 - (miz - cz)**2) - r)**2

        fdcx = diff(fcx,cxs)

        cys = symbols('cys')
        fcy = (sqrt((mix - cx)**2 - (miy - cys)**2 - (miz - cz)**2) - r)**2

        fdcy = diff(fcy,cys)

        czs = symbols('czs')
        fcz = (sqrt((mix - cx)**2 - (miy - cy)**2 - (miz - czs)**2) - r)**2

        fdcz = diff(fcz,czs)

        rs = symbols('rs')
        fradius = (sqrt((mix - cx)**2 - (miy - cy)**2 - (miz - cz)**2) - rs)**2

        fdradius = diff(fradius,rs)
        jacobian_sphere = np.array([fdcx, fdcy, fdcz, fdradius])

        print(jacobian_sphere,'jacobian_sphere2')

        return jacobian_sphere

    def testing(self, input_vector):

        jacobian_sphere = []

        print(np.shape(input_vector),'input_vector')

        for k in range(len(input_vector)):
            jacobian_sphere.append(self.ThreadPool.map(self.compute_jacobian_sphere,input_vector[k]))

        return jacobian_sphere

    def wtf(self, input_vector):

        jacobian_sphere = s.testing(np.transpose(input_vector))

        return jacobian_sphere

if __name__ == '__main__':

    print(sqrt(0.0207058869923876),'0.0207058869923876')

    exit()

    q = np.array([1,2,3])
    w = np.array([1,2,3])
    e = np.array([1,2,3])

    # a = Matrix([q,w,e])
    #
    # q, w, e = symbols('q, w, e')
    #
    #
    # b = Matrix([q, w, e])

    t = np.array([[1,1,1],[1,1,1],[1,1,1]])

    r = np.array([1,1,0])

    print(np.dot(r,t),'dot______')

    exit()

    b = np.array([q, w ,e])

    a = [-0.0215724313324132, -2.39812678811891, 1.85915552334908]

    print(a,'a')
    print(np.transpose(a),'a')
    print(np.linalg.norm(np.transpose(a)),'a')

    print(math.sqrt( np.linalg.norm(b[0])**2 + np.linalg.norm(b[1])**2 + np.linalg.norm(b[2])**2  ),'b')

    exit()

    c = np.array([q,w,e])

    d = np.array([1,2,3])

    print(shape(c),'c')
    print(shape(d),'d')

    print(shape(a),'a')
    print(shape(b),'b')

    print(d*c,' ?? ')

    print(b*a,'b*a')

    exit()

    print(shape(b),'b shape')

    e = Matrix([5])

    print(shape(e),'e')

    print(b.add(e) ,'b+e')


    exit()

    print(shape(c),shape(d))
    print(shape(c.T))

    print(c.T,'c.T')

    print(c.T - d,'c-d')

    print(shape(a))
    print(shape(b))
    print(a,'a')
    print(b,'b')
    # print(shape(b*a))
    print(a - b ,'a - b ')

    exit()

    s = some()

    mix = 0.089924
    miy = 0.101644
    miz = 0.535250
    cx = 0.099813
    cy = 0.066042
    cz = 0.529096
    r = -0.018959

    mixs,miys,mizs,cxs,cys,czs,rs = symbols('mixs,miys,mizs,cxs,cys,czs,rs')

    x = 2*(cxs - mixs)*(-rs + sqrt((-cxs + mixs)**2 + (-cys + miys)**2 + (-czs + mizs)**2))/sqrt((-cxs + mixs)**2 + (-cys + miys)**2 + (-czs + mizs)**2)
    fdcx = lambdify([mixs,miys,mizs,cxs,cys,czs,rs],x)
    fdcx = fdcx(mix,miy,miz,cx,cy,cz,r)

    print(fdcx,'fdcx')
    exit()
    # f = implemented_function('f', lambda x,y: x+1+y)
    #
    # lam_f = lambdify(x, f(x,y))
    #
    # print(lam_f(4,1),'lam_f(4)')

    mixs = symbols('mixs')

    f = lambdify([mixs,y,z],mixs+y+z)

    f(1,1,1)

    print(f(1,1,1),'f(1,1,1)')

    exit()

    # b = s.startingthing()
    #
    # print(b,'yes')



    # input_vector = []
    #
    # for i in range(2):
    #     input_vector.append(np.loadtxt('camera/jac_input_vector'))
    #
    # print(len(input_vector),'input_vector')
    # print(len(input_vector[0]),'input_vector')
    #
    # jacobian_sphere = s.testing(input_vector)
    #
    # # print(jacobian_sphere,'jacobian_sphere')
    # print(np.shape(jacobian_sphere),'jacobian_sphere')
