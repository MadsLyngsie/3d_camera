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
from sympy import *
from sympy.vector import *
import functools
from multiprocessing.pool import ThreadPool
from sympy.utilities.lambdify import lambdify, implemented_function
# from sympy.abc import x

class some:
    def __init__(self):
        self.ThreadPool = ThreadPool(processes = 100)
        self.len = 0
        self.a = 0
        self.mix = 0
        self.miy = 0
        self.miz = 0
        self.deriv_cx = 0
        self.deriv_cy = 0
        self.deriv_cz = 0
        self.deriv_r  = 0


    def compute_jacobian_sphere(self, input_vector):

        self.len = len(input_vector)

        self.mix = input_vector[0]
        self.miy = input_vector[1]
        self.miz = input_vector[2]
        cx = input_vector[3]
        cy = input_vector[4]
        cz = input_vector[5]
        r = input_vector[6]


        jacobian = self.jacobian_sphere(cx,cy,cz,r)

        return jacobian

    def jacobian_sphere(self,cx,cy,cz,r):

        mixs,miys,mizs,cxs,cys,czs,rs = symbols('mixs,miys,mizs,cxs,cys,czs,rs')

        fdcx = lambdify([mixs,miys,mizs,cxs,cys,czs,rs],self.deriv_cx)
        fdcx = fdcx(self.mix,self.miy,self.miz,cx,cy,cz,r)

        mixs,miys,mizs,cxs,cys,czs,rs = symbols('mixs,miys,mizs,cxs,cys,czs,rs')

        fdcy = lambdify([mixs,miys,mizs,cxs,cys,czs,rs],self.deriv_cy)
        fdcy = fdcy(self.mix,self.miy,self.miz,cx,cy,cz,r)

        mixs,miys,mizs,cxs,cys,czs,rs = symbols('mixs,miys,mizs,cxs,cys,czs,rs')

        fdcz = lambdify([mixs,miys,mizs,cxs,cys,czs,rs],self.deriv_cz)
        fdcz = fdcz(self.mix,self.miy,self.miz,cx,cy,cz,r)

        mixs,miys,mizs,cxs,cys,czs,rs = symbols('mixs,miys,mizs,cxs,cys,czs,rs')

        fdradius = lambdify([mixs,miys,mizs,cxs,cys,czs,rs],self.deriv_r)
        fdradius = fdradius(self.mix,self.miy,self.miz,cx,cy,cz,r)

        jacobian_sphere = [fdcx, fdcy, fdcz, fdradius]

        return jacobian_sphere

    def testing(self, input_vector):

        jacobian_sphere = []

        mixs,miys,mizs,cxs,cys,czs,rs = symbols('mixs,miys,mizs,cxs,cys,czs,rs')

        fcx = sqrt((mixs - cxs)**2 + (miys - cys)**2  + (mizs - czs)**2) - rs
        fcy = sqrt((mixs - cxs)**2 + (miys - cys)**2  + (mizs - czs)**2) - rs
        fcz = sqrt((mixs - cxs)**2 + (miys - cys)**2  + (mizs - czs)**2) - rs
        fradius = sqrt((mixs - cxs)**2 + (miys - cys)**2  + (mizs - czs)**2) - rs

        self.deriv_cx = diff(fcx,cxs)
        self.deriv_cy = diff(fcy,cys)
        self.deriv_cz = diff(fcz,czs)
        self.deriv_r  = diff(fradius,rs)

        # print(self.deriv_cx,'self.deriv_cx')

        # print(input_vector,'input_vector')

        jacobian_sphere = (self.ThreadPool.map(self.compute_jacobian_sphere,input_vector))
        # print(self.a,'self.a')
        # print(jacobian_sphere,'jacobian_sphere')

        return jacobian_sphere


if __name__ == '__main__':

    s = some()
    input_vector = []

    input_vector = (np.loadtxt('camera/jac_input_vector'))


    print(len(input_vector),'input_vector')
    print(np.shape(input_vector),'input_vector')

    jacobian_sphere = s.testing(np.transpose(input_vector))

    print(jacobian_sphere,'jacobian_sphere')
    print(np.shape(jacobian_sphere),'jacobian_sphere')
