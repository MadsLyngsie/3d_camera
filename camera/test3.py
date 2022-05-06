import pyrealsense2 as rs
import cv2
import numpy as np
import open3d as o3d
import math
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
import functools
from multiprocessing.pool import ThreadPool
from numpy.linalg import inv
import time
import sympy as sp
from sympy.utilities.autowrap import autowrap
from sympy.vector import CoordSys3D
from SO3 import SO3
from SE3 import SE3


mesh_cylinder  = o3d.geometry.TriangleMesh.create_cylinder(radius = 0.025, height = 0.2, resolution = 100)

points = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh_cylinder,100)

print(mesh_cylinder.PointCloud,'mesh_cylinder')
print(np.asarray(points.points),'points')
