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


class some:
    def __init__(self):
        self.ThreadPool = ThreadPool(processes = 100)
        self.len = 0

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
        fcx = (1/self.len)*(sqrt((mix - cxs)**2 - (miy - cy)**2 - (miz - cz)**2) - r)**2

        fdcx = diff(fcx,cxs)

        cys = symbols('cys')
        fcy = (1/self.len)*(sqrt((mix - cx)**2 - (miy - cys)**2 - (miz - cz)**2) - r)**2

        fdcy = diff(fcy,cys)

        czs = symbols('czs')
        fcz = (1/self.len)*(sqrt((mix - cx)**2 - (miy - cy)**2 - (miz - czs)**2) - r)**2

        fdcz = diff(fcz,czs)

        rs = symbols('rs')
        fradius = (1/self.len)*(sqrt((mix - cx)**2 - (miy - cy)**2 - (miz - cz)**2) - rs)**2

        fdradius = diff(fradius,rs)
        jacobian_sphere = np.array([fdcx, fdcy, fdcz, fdradius])

        return jacobian_sphere

    def testing(self, input_vector):

        jacobian_sphere = self.ThreadPool.map(self.compute_jacobian_sphere,input_vector)

        return jacobian_sphere

    def wtf(self, input_vector):

        jacobian_sphere = s.testing(np.transpose(input_vector))

        return jacobian_sphere

if __name__ == '__main__':

    s = some()

    input_vector = np.loadtxt('camera/jac_input_vector')

    print(len(input_vector),'input_vector')
    print(len(input_vector[0]),'input_vector')

    jacobian_sphere = s.wtf(input_vector)

    print(jacobian_sphere,'jacobian_sphere')
    print(len(jacobian_sphere),'jacobian_sphere')
