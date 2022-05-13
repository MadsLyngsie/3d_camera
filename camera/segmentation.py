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

class Segmentation:

    def __init__(self):
        self.ThreadPool = ThreadPool(processes = 100)
        self.len = 0
        self.mix = 0
        self.miy = 0
        self.miz = 0
        self.cxc = 0
        self.cyc = 0
        self.czc = 0
        self.rc = 0
        self.jacobian_sphere_input_vector = 0
        self.deriv_cx = 0
        self.deriv_cy = 0
        self.deriv_cz = 0
        self.deriv_r  = 0


    def filter_neighbors(self, xyz, neighbors):
        #neighbors filtering
        tree = KDTree(xyz, leaf_size = 200)
        dist, ind = tree.query(xyz, k =neighbors)
        #compute meandist
        meandist = np.mean(dist, axis=1)

        #computes the standard deviation of mean distance space
        mdstd = np.std(meandist)

        #computes mean of mean distance space
        mdmean = np.mean(meandist)

        #compute the min and max range for the filter
        alpha = 1
        minpxyz = (mdmean - alpha * mdstd)
        maxpxyz = (mdmean + alpha * mdstd)

        #filter the PC with meandist
        inliers  = np.where((meandist > minpxyz ) & (meandist < maxpxyz))

        #Matching of index to pcd
        xyz = xyz[inliers]

        return xyz

    def filter_radius(self,xyz,radius,minpoints,visualize):

        #tree query to find radius of of nearest points
        tree = KDTree(xyz, leaf_size = 200)
        ind = tree.query_radius(xyz, r=radius)
        no_pts_in_given_radius = np.array([len(ind[i]) for i in range(ind.shape[0])])

        #points that should stay since there is enough points points close
        inliers_radius  = np.where((no_pts_in_given_radius > minpoints))

        #keeping inliers
        xyz = xyz[inliers_radius]

        #visualization of point cloud with neighbor and radius filtering
        if visualize == True:
            self.pcd = o3d.geometry.PointCloud()

            self.pcd.points  = o3d.utility.Vector3dVector(xyz)

            #visualize with surface normals from open3d
            #self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

            o3d.visualization.draw_geometries( [self.pcd], point_show_normal=True)

        return xyz

    def surface_normal_estimation(self, xyz, neighbors, test):

        #tree query for neighbors
        tree = KDTree(xyz)
        dist, ind = tree.query(xyz, k = neighbors)

        #make centorid indices
        Xi = xyz[ind]

        #multiprocessing the calculation of the normals
        normals = p.map(seg.normals,Xi)


        #test to show normals calculated maunally
        if test == True:
            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points  = o3d.utility.Vector3dVector(xyz)
            self.pcd.normals = o3d.utility.Vector3dVector(normals)
            o3d.visualization.draw_geometries( [self.pcd], point_show_normal=True)

        return normals

    @staticmethod
    def normals(Xi):

        #calculatetion of nomals
        xm = np.mean(Xi,axis = 0)
        neighbors = len(Xi)

        cov = (1/neighbors)*( np.dot(np.transpose(Xi-xm), (Xi - xm)))

        w,v = LA.eig(cov)

        #sort normals
        idx = w.argsort()
        w = w[idx]

        v = v[:,idx]

        #keep the smalles normals
        normals = v[:,0]

        #flip normals so they are all in the same direction
        if normals.dot(xm)>0:
            normals = -normals

        return  normals

    @staticmethod
    def normals_max(Xi):

        xm = np.mean(Xi,axis = 0)
        cov = np.cov(Xi)
        #print(Xi,'xi')
        #print(cov,'cov')

        w,v = LA.eig(cov)

        #sort normals
        idx = w.argsort()
        w = w[idx]

        v = v[:,idx]

        #keep the biggest normals
        normals = w[:,1]

        #print(w,'w_max')

        # #flip normals so they are all in the same direction
        # if normals.dot(xm)>0:
        #     normals = -normals

        return  normals

    @staticmethod
    def normals_min(Xi):

        xm = np.transpose(np.mean(Xi,axis = 0))
        cov = np.cov(Xi)

        w,v = LA.eig(cov)

        #sort normals
        idx = w.argsort()
        w = w[idx]

        v = v[:,idx]

        #keep the smalles normals
        normals = w[:,0]

        #print(w,'w_min')

        # #flip normals so they are all in the same direction
        # if normals.dot(xm)>0:
        #     normals = -normals

        return  normals

    def gaussian_image(self, normals):

        #show the guassian image of picture in 3d plot
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        #puts normals in np array
        normals = np.array(normals)

        #compute the scatterplot of nomrals
        ax.scatter3D(normals[:,0],normals[:,1],normals[:,2], c=normals[:,2], cmap='Greens')

        #set labels to understand plot
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)
        plt.show()

        return

    def crease_removal(self,xyz,normals,neighbors):

        #the points the spanc a crease are found by doing a cos sim of the centroid
        # if the cos sim is above a treshhold the points are deemed to not span a crease
        # if the the cos sim is 1 all pints are in a plane.

        #tree search of neighbors
        tree = KDTree(xyz)
        dist, ind = tree.query(xyz, k = neighbors)
        Xi = xyz[ind]

        #make sure normals and np array
        normals= np.array(normals)

        #make empty array for cos sim and points that does not span a crease
        Cosine_Similarity =  np.zeros((len(normals), 30))
        not_creaseind = np.zeros(len(normals))

        #computes the cos sim
        for i in range(len(normals)):
            anorm = np.array(normals[ind[i]])
            bnorm = np.array(normals[i])

            Cosine_Similarity[i] = np.dot(anorm, bnorm)

            temp_index = np.where(Cosine_Similarity[i] > 0.75) ## 0.75 is good

            #keep points that does not span a crease
            if len(temp_index[0]) >  29:
                not_creaseind[i] = int(i)

        #point cloud with creases removed
        xyz = xyz[not_creaseind.astype(int)]

        return xyz

    def segmentate(self, xyz, normals, visualize):

        #doing a dbscan to attemp a clustering of points in the pcd
        db = DBSCAN(eps=0.02 , min_samples=55).fit(xyz) # 0.022 & 50
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        ##print(labels,'labels')
        labels_unq = np.unique(labels)
        ##print(labels_unq,'labels_unq')

        #save the clusters and the noise from the scan
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)


        #visualize the clusters
        if visualize == True:
            fig = plt.figure()

            ax = plt.axes(projection='3d')


            scatter = ax.scatter3D(xyz[:,0],xyz[:,1],xyz[:,2], c=db.labels_, label = labels_unq,  cmap='jet')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            # legend = ax.legend(scatter.legend_elements(num=len(labels_unq)), loc="upper left", title="Ranking")

            legend = ax.legend(*scatter.legend_elements(), loc = "lower left", title = 'classes')

            ax.add_artist(legend)

            ax.view_init(elev=-20., azim=280)
            plt.savefig("clustering")

        return labels

    def bin_removal(self,xyz, read_from_file, rgb_img, draw, draw2d):
        #this function removes the points of the bin
        #this is done by est the pose of a aruco marker and useing the size of the bin
        #the pound cloud data points are transformed to be seen from the bin and
        #then compared to the length width and height of the bin

        #define marker finder parameters and find marker
        marker_size = 0.105 #size of marker in M
        type_arucodict = cv2.aruco.DICT_ARUCO_ORIGINAL
        arucoDict = cv2.aruco.Dictionary_get(type_arucodict)
        arucoParams = cv2.aruco.DetectorParameters_create()

        if read_from_file == False:
            (corners, ids, rejected) = cv2.aruco.detectMarkers(rgb_img, arucoDict, parameters=arucoParams)

            #est the pose of the marker
            frame_markers = cv2.aruco.drawDetectedMarkers(rgb_img.copy(), corners, ids)
            if len(corners) > 0:
                ids = ids.flatten()

                for i in range(np.shape(corners)[0]):
                    #estimation of aruco pose
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], marker_size, cam.camera_mat, cam.distCoeffs)
                    frame_markers = cv2.drawFrameAxes(frame_markers, cam.camera_mat, cam.distCoeffs, rvecs, tvecs, length=0.05, thickness=2)

                    #constroction of homugeous transformatiob matrix
                    rotmat, _ = cv2.Rodrigues(rvecs)
                    tvecs = np.reshape(tvecs, (3, 1))
                    cam2marker = np.concatenate((rotmat, tvecs), axis = 1)
                    cam2markerHom = np.concatenate((cam2marker, np.array([[0, 0, 0, 1]])), axis = 0)

                    #draw line from the found marker
                    if draw == True:
                        cv2.imshow("Image",frame_markers)
                        cv2.waitKey(1)

                    np.savetxt('camera/frame3.txt', cam2markerHom)
        if read_from_file == True:
            cam2markerHom = np.loadtxt('camera/frame3.txt')


        #define the markers corner in bin the bin frame
        marker_bin_corner = np.array([[-marker_size/2],[marker_size/2],[0],[1]])

        #construct homugeous transformation matrix for 2d
        hom_trans_to_marker_corner = np.zeros((4,4))
        np.fill_diagonal(hom_trans_to_marker_corner,1)
        hom_trans_to_marker_corner[0,3] = marker_bin_corner[0][0]
        hom_trans_to_marker_corner[1,3] = marker_bin_corner[1][0]
        hom_trans_to_marker_corner[2,3] = marker_bin_corner[2][0]

        #make pcd homugeous
        row = np.ones(len(xyz))
        xyz1 = np.concatenate((np.transpose(xyz), np.array([row])), axis = 0)

        #move the pcd to the
        markerpcd = np.dot(cam2markerHom,xyz1)

        ##############################################
        # # original
        bin_width    =  0.34 #M real is 0.34 #0.34 original
        bin_length   =  0.40 #M real is 0.44 #0.44 original
        bin_height   =  0.3 #M real is 0.2 #0.5 original

        # bin_width    =  0.3
        # bin_length   =  0.415
        # bin_height   =  0.5

        # orignal
        condt = np.where((markerpcd[0] >= 0.05  )  & (markerpcd[0] <= bin_width) &
                         (markerpcd[1] <= -0.05) & (markerpcd[1] >= -bin_length) &
                         (markerpcd[2] >= 0.075) & (markerpcd[2] <= bin_height) )

        # condt = np.where((markerpcd[0] >= -0.375)  & (markerpcd[0] <= bin_width) &
        #                  (markerpcd[1] <= 0.4) & (markerpcd[1] >= -bin_length) &
        #                  (markerpcd[2] >= -0.2) & (markerpcd[2] <= bin_height) )

        # #print(markerpcd,'markerpcd')

        ##############################################
        # # Best 2
        # bin_width    =  0.28 #M real is 0.34 #0.22
        # bin_length   =  0.47 #M real is 0.44 #0.39
        # bin_height   =  0.75 #M real is 0.2 #0.175
        #
        # condt = np.where((markerpcd[0] >= -0.05)  & (markerpcd[0] <= bin_width) &
        #                  (markerpcd[1] <= -0.05) & (markerpcd[1] >= -bin_length) &
        #                  (markerpcd[2] >= 0.08) & (markerpcd[2] <= bin_height) ) ## 0.19 is really good

        xyz = xyz[condt]


        if draw2d == True:
            ########show 2d circle on the corner of the marker useing the 3d pose
            bin_corner = np.dot(cam2markerHom,marker_bin_corner)
            bin_corner_2d = (np.dot(cam.camera_mat , bin_corner[:3])/bin_corner[2][0])[:2]
            cv2.circle(rgb_img, (int(bin_corner_2d[0]),int(bin_corner_2d[1])), radius=5, color=(0, 0, 255), thickness=2)
            cv2.imshow("Image1",rgb_img)
            cv2.waitKey(1)

        return xyz

    def non_parametric_surface_class(self, xyz, clusterlabels, neighbors):

        ##### non_parametric_surface_classification useing the method outlined in
        # visal perception and robotic manipulation 3d chapter 4.4

        ### can be # OPTIMIZE: for loop into vector notation
        clusters = []
        cluster_idx = []
        cluster_mean = []
        ni = []
        mi = []
        principal_norals = []
        Cosine_Similarity_primitive = []
        shape_guess = []
        shape_guess_eig = []
        shape_guess_procent = []
        convexity_guess = []
        convexity_guess_procent = []
        uni_conxeity = []
        surface_class = []
        surface_class_eig = []
        labels_unq = np.unique(clusterlabels)


        ### loop to look at each cluter 1 by 1
        for i in range(np.amax(clusterlabels)+1):

            #print(i,'i')

            #get the index of the cluster that is being classified
            cluster_idx.append(np.where((clusterlabels == i)))

            ## empty arrat for clusters
            clusters = []

            ## extract cluster from pcd
            clusters.append(xyz[cluster_idx[i]])

            ## tree qurry over the current cluster
            tree = KDTree(np.squeeze(clusters))
            dist, ind = tree.query(np.squeeze(clusters), k = neighbors)

            clusters_ind = np.squeeze(clusters)[ind]

            ## multi processing of normals for current cluster
            cluster_normals = np.asarray(p.map(seg.normals,clusters_ind))

            ## calculate normal centroid
            ni = np.mean(cluster_normals,axis = 0)
            ni = ni/np.linalg.norm(ni)

            ## initialize z axis
            z_hat = np.array([0,0,-1])

            #compute the alignment transofrmation with the z axis
            r_theta = np.arccos(np.dot(ni,z_hat))
            r_A = np.cross(ni,z_hat)

            ## Rodrigues formulation for rmat
            skew_w = np.array([[0,-r_A[2],r_A[1]],[r_A[2],0,-r_A[0]],[-r_A[1],r_A[0],0]])

            r_mat = np.eye(3) + skew_w + (skew_w.dot(skew_w))/(1+np.cos(r_theta))

            ## normals of cluster aligned with z axis
            nj_alligned = np.dot(np.squeeze(r_mat),np.transpose(cluster_normals))


            ## compute aligend normal centroid
            nja = np.mean(nj_alligned,axis = 1)
            nja = nja/np.linalg.norm(nja)


            ## decompose the current cluster for better computation
            decomp_cluster = np.transpose(nj_alligned)
            neighbors_decomp = 150

            if (len(decomp_cluster) < neighbors_decomp):
                neighbors_decomp = len(decomp_cluster)

            #print(decomp_cluster,'decomp_cluster')

            ## tree search to make smaller patches for robutsness of computation
            tree = KDTree(decomp_cluster)
            dist, ind = tree.query(np.squeeze(decomp_cluster), k = neighbors_decomp)

            decomp_cluster = np.asarray(decomp_cluster[ind])

            ## initialize variables for decomposed cluster loop
            cluster_concl = []
            cluster_concl_eig = []
            Np = []
            ## 0.000056 5.6e-05
            ## 0.000071
            ## v0.000033
            eth = 0.000675
            eth_eig = 0.025
            #print(eth,'eth')
            unique = []
            counts = []
            mse_max_app = []
            mse_min_app = []
            w_comp = []
            convex_guess = 0
            concave_guess = 0
            neither_guess = 0
            convex = 0
            concave = 0
            np_eig = 0

            mi = np.mean(clusters,axis = 1)
            mi = mi/np.linalg.norm(mi)


            for j in range(len(decomp_cluster[0])):
                ############### Principal Curvatures ####################

                decomp_cluster_temp = decomp_cluster[j][:]
                m = np.sqrt(len(decomp_cluster_temp))

                ## quardratic solution
                delta = np.exp2(np.sum(np.exp2(decomp_cluster_temp[0,:]) - \
                                       np.exp2(decomp_cluster_temp[1,:]))) + \
                                       4 * (np.exp2(np.sum( decomp_cluster_temp[0,:] * \
                                                           decomp_cluster_temp[1,:])))

                ## compute max and min angle for quardratic solution
                theta_max = np.arctan((np.sum(np.exp2(decomp_cluster_temp[1,:]) - \
                                               np.exp2(decomp_cluster_temp[0,:])) + \
                                               np.sqrt(delta)) /\
                                       (2 * np.sum(decomp_cluster_temp[0,:] * (decomp_cluster_temp[1,:]))))

                theta_min = np.arctan2((np.sum(np.exp2(decomp_cluster_temp[1,:]) -\
                                               np.exp2(decomp_cluster_temp[0,:])) - \
                                               np.sqrt(delta)) ,\
                                       (2 * np.sum(decomp_cluster_temp[0,:] * (decomp_cluster_temp[1,:]))))

                ## max and min mean square error
                mse_max = (1/np.exp2(m)) * \
                          np.sum(np.exp2( (decomp_cluster_temp[0,:] * np.cos(theta_max)) + \
                                          (decomp_cluster_temp[1,:] * np.sin(theta_max)) ))

                mse_min = (1/np.exp2(m)) * \
                          np.sum(np.exp2( (decomp_cluster_temp[0,:] * np.cos(theta_min)) + \
                                          (decomp_cluster_temp[1,:] * np.sin(theta_min)) ))

                mse_max_app.append(mse_max)
                mse_min_app.append(mse_min)

                ## find number of non principal curvatures
                if ((mse_max < eth) and (mse_min < eth)):
                    Np = 0
                elif ((mse_max < eth) and (mse_min > eth)):
                    Np = 1
                elif ((mse_max > eth) and (mse_min < eth)):
                    Np = 1
                elif ((mse_max > eth) and (mse_min > eth)):
                    Np = 2

                cluster_concl.append(Np)

                ############### convexity ####################

                ## dj is a vector pointing from the centorid normal to a surface element
                dj = np.transpose(np.array(decomp_cluster[j])) - np.transpose(np.array(mi))

                ## compute convexity of demcomp patch
                convexity = np.dot( np.squeeze(np.cross(np.cross(cluster_normals[j],ni),np.transpose(dj))) ,np.vstack(ni))

                ## add convexity up for all normals in the patch
                convex = 0
                concave = 0
                for k in range(len(convexity)):

                    if (convexity[k] >= 0):
                        convex += 1
                    else:
                        concave += 1
                ## makeing sure s is not calculated by 0 division
                if (convex == 0):
                    s = concave
                elif (concave == 0):
                    s = convex
                else:
                    s = convex / concave
                ## sth proposed in paper
                sth = 1.5

                ## convexity classified for decomposed patch
                if (s>sth):
                    convex_guess += 1
                elif (s<(1/sth)):
                    concave_guess += 1
                elif (((1/sth)<s) and (s<sth)):
                    neither_guess += 1

                ############ eig Np est #####################

                ## computing the normals for eig value metohd of decideing Np
                ## find mean of demcomp patches
                xm = np.squeeze(np.mean(decomp_cluster,axis = 0))
                ## covmat of decompose patch
                cov = (1/len(decomp_cluster))*( np.dot(np.transpose((decomp_cluster[j])-xm), (decomp_cluster[j] - xm)))

                ## np eig fucntion
                w,v = LA.eig(cov)

                #sort normals
                idx = w.argsort()
                w = w[idx]

                np_eig = 0

                ## delete middel value from eig vaules
                w = np.delete(w, 1)

                ## count np
                for k in range(len(w)):
                    if (w[k] > eth_eig):
                        np_eig += 1

                cluster_concl_eig.append(np_eig)
                ##############################################

            ## printing information about results
            #print(convex_guess,'convex_guess')
            #print(concave_guess,'concave_guess')
            #print(neither_guess,'neither_guess')
            #print(np.mean(mse_max_app),'mse_max_app')
            #print(np.mean(mse_min_app),'mse_min_app')

            ## universal convexity
            uni_conxeity = []
            uni_conxeity.append(convex_guess)
            uni_conxeity.append(concave_guess)
            uni_conxeity.append(neither_guess)

            ## count np's and convexity of decomped patches
            unique_uni_conxeity, counts_uni_conxeity = np.unique(uni_conxeity, return_counts=True)
            unique, counts = np.unique(cluster_concl, return_counts=True)
            unique_eig, counts_eig = np.unique(cluster_concl_eig, return_counts=True)

            ## printing information about results
            #print(unique,'unique')
            #print(counts,'counts')
            #print(unique_eig,'unique_eig')
            #print(counts_eig,'counts_eig')
            #print(counts_uni_conxeity,'counts_uni_conxeity')

            ## accturl guesses for current clusters
            convexity_guess.append(uni_conxeity.index(np.amax(uni_conxeity)))
            shape_guess.append(np.array(unique[np.where(counts == np.amax(counts))]))
            shape_guess_eig.append(np.array(unique_eig[np.where(counts_eig == np.amax(counts_eig))]))

            ## procent confidence on current shape/ convexity guess
            convexity_guess_procent.append(np.amax(uni_conxeity)/np.sum(uni_conxeity))
            shape_guess_procent.append(np.amax(counts)/np.sum(counts))
            shape_guess_procent.append(np.amax(counts_eig)/np.sum(counts_eig))

            ################ classes ####################
            ## table of how to understand surface_class information and surface_class_eig
            # plane   = 0
            # ridge   = 1
            # peak    = 2
            # valley  = 3
            # pit     = 4
            # saddle  = 5

            #print(np.squeeze(shape_guess),'shape_guess')

            ## putting guess into results
            if ( (shape_guess[i] == 0) and (convexity_guess[i] == 0) ):
                surface_class.append(0)
            elif ( (shape_guess[i] == 0) and (convexity_guess[i] == 1) ):
                surface_class.append(0)
            elif ( (shape_guess[i] == 0) and (convexity_guess[i] == 2) ):
                surface_class.append(0)
            elif ( (shape_guess[i] == 1) and (convexity_guess[i] == 0) ):
                surface_class.append(1)
            elif ( (shape_guess[i] == 1) and (convexity_guess[i] == 1) ):
                surface_class.append(3)
            elif ( (shape_guess[i] == 1) and (convexity_guess[i] == 2) ):
                surface_class.append(5)
            elif ( (shape_guess[i] == 2) and (convexity_guess[i] == 0) ):
                surface_class.append(2)
            elif ( (shape_guess[i] == 2) and (convexity_guess[i] == 1) ):
                surface_class.append(4)
            elif ( (shape_guess[i] == 2) and (convexity_guess[i] == 2) ):
                surface_class.append(5)

            ## putting guess into results for eig method
            if ( (shape_guess_eig[i] == 0) and (convexity_guess[i] == 0) ):
                surface_class_eig.append(0)
            elif ( (shape_guess_eig[i] == 0) and (convexity_guess[i] == 1) ):
                surface_class_eig.append(0)
            elif ( (shape_guess_eig[i] == 0) and (convexity_guess[i] == 2) ):
                surface_class_eig.append(0)
            elif ( (shape_guess_eig[i] == 1) and (convexity_guess[i] == 0) ):
                surface_class_eig.append(1)
            elif ( (shape_guess_eig[i] == 1) and (convexity_guess[i] == 1) ):
                surface_class_eig.append(3)
            elif ( (shape_guess_eig[i] == 1) and (convexity_guess[i] == 2) ):
                surface_class_eig.append(5)
            elif ( (shape_guess_eig[i] == 2) and (convexity_guess[i] == 0) ):
                surface_class_eig.append(2)
            elif ( (shape_guess_eig[i] == 2) and (convexity_guess[i] == 1) ):
                surface_class_eig.append(4)
            elif ( (shape_guess_eig[i] == 2) and (convexity_guess[i] == 2) ):
                surface_class_eig.append(5)


            ## printing final results and stats about the performance.
            #print(np.squeeze(shape_guess),'shape_guess')
            #print(np.squeeze(shape_guess_eig),'shape_guess_eig')
            #print(np.squeeze(convexity_guess),'convexity_guess')
            #print(np.squeeze(shape_guess_procent),'shape_guess_procent')
            #print(np.mean(np.squeeze(shape_guess_procent)),'shape_guess_procent_mean')
            #print(np.squeeze(convexity_guess_procent),'convexity_guess_procent')
            #print((np.mean(np.squeeze(convexity_guess_procent)) + np.mean(np.squeeze(shape_guess_procent)))/2,'mean mean')
            #print(surface_class,'surface_class')
            #print(surface_class_eig,'surface_class_eig')
        return surface_class, surface_class_eig

    def jacobian_sphere(self,cx,cy,cz,r):

        mixs,miys,mizs,cxs,cys,czs,rs = symbols('mixs,miys,mizs,cxs,cys,czs,rs')

        # check different lambdify options and comparwe speed
        # map faster if lambdify output is C for loop is faster

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

    def compute_jacobian_sphere_loop(self, lambdified_jacobian_sphere, input_vector):
        result = lambdified_jacobian_sphere(*input_vector).reshape((4,)).astype('float32')
        return result

    # def execute_jacobian(self, input_vector):

        jacobian_sphere = []

        mixs,miys,mizs,cxs,cys,czs,rs = symbols('mixs,miys,mizs,cxs,cys,czs,rs')

        e_sph = sqrt((mixs - cxs)**2 + (miys - cys)**2  + (mizs - czs)**2) - rs
        # fcy = sqrt((mixs - cxs)**2 + (miys - cys)**2  + (mizs - czs)**2) - rs
        # fcz = sqrt((mixs - cxs)**2 + (miys - cys)**2  + (mizs - czs)**2) - rs
        # fradius = sqrt((mixs - cxs)**2 + (miys - cys)**2  + (mizs - czs)**2) - rs

        self.deriv_cx = diff(e_sph,cxs)
        self.deriv_cy = diff(e_sph,cys)
        self.deriv_cz = diff(e_sph,czs)
        self.deriv_r  = diff(e_sph,rs)
        jacobian_sphere = (self.ThreadPool.map(self.compute_jacobian_sphere,input_vector))

        return jacobian_sphere

    def lambdifing_jacobian_sphere(self):

        jacobian_sphere = []

        mixs,miys,mizs,cxs,cys,czs,rs = symbols('mixs,miys,mizs,cxs,cys,czs,rs')

        e_sph = sqrt((mixs - cxs)**2 + (miys - cys)**2  + (mizs - czs)**2) - sp.sign(rs)*rs

        deriv_cx = diff(e_sph,cxs)
        deriv_cy = diff(e_sph,cys)
        deriv_cz = diff(e_sph,czs)
        deriv_r  = diff(e_sph,rs)

        lambdified_jacobian_sphere = sp.Matrix([deriv_cx,deriv_cy,deriv_cz,deriv_r])
        lambdified_jacobian_sphere  = lambdified_jacobian_sphere.subs(sp.diff(sp.sign(rs), rs), 0)
        lambdified_jacobian_sphere  = autowrap(lambdified_jacobian_sphere, backend="f2py", args = [mixs,miys,mizs,cxs,cys,czs,rs])

        return lambdified_jacobian_sphere

    def lambdifing_jacobian_cylinder(self):

        jacobian_cylinder = []

        d, phi, theta, alpha, mix, miy, miz, r = symbols('d, phi, theta, alpha, mix, miy, miz, r')

        vector_n = sp.Matrix([[sp.cos(phi) * sp.sin(theta) ], \
                              [sp.sin(phi) * sp.sin(theta) ], \
                              [sp.cos(theta)                    ]])

        # #print(vector_n,'vector_n')

        vector_n_theta = sp.Matrix([[sp.cos(phi) * sp.cos(theta) ], \
                                    [sp.sin(phi) * sp.cos(theta) ], \
                                    [-sp.sin(theta)                   ]])
        # #print(vector_n_theta,'vector_n_theta')

        vector_n_phi = sp.Matrix([[-sp.sin(phi)   ], \
                                  [sp.cos(phi)    ], \
                                  [0                   ]])

        # #print(vector_n_phi,'vector_n_phi')

        vector_a = sp.Matrix(vector_n_theta * sp.cos(alpha) + vector_n_phi * sp.sin(alpha))

        # #print(vector_a,'vector_a')

        vector_mi = sp.Matrix([ mix, miy, miz ])

        # #print(vector_mi,'vector_mi')

        vector_p = sp.Matrix(d*vector_n)

        # #print(vector_p,'vector_p')


        e_cylinder0 = ((vector_p + vector_a * (vector_mi - vector_p).T * vector_a - vector_mi)[0])**2
        e_cylinder1 = ((vector_p + vector_a * (vector_mi - vector_p).T * vector_a - vector_mi)[1])**2
        e_cylinder2 = ((vector_p + vector_a * (vector_mi - vector_p).T * vector_a - vector_mi)[2])**2

        e_cylinder = sqrt(e_cylinder0 + e_cylinder1 + e_cylinder2) - sp.sign(r)*r

        # e_cylinder = (vector_p + vector_a * (vector_mi - vector_p).T * vector_a - vector_mi) - sp.sign(r)*r

        # #print(e_cylinder,'e_cylinder')
        # #print(shape(e_cylinder),'e_cylinder')


        deriv_d = diff(e_cylinder,d)
        deriv_theta = diff(e_cylinder,theta )
        deriv_phi = diff(e_cylinder,phi)
        deriv_alpha = diff(e_cylinder,alpha)
        deriv_r = diff(e_cylinder,r)

        lambdified_jacobian_cylinder  = sp.Matrix([deriv_d, deriv_theta, deriv_phi, deriv_alpha, deriv_r])
        lambdified_jacobian_cylinder  = lambdified_jacobian_cylinder.subs(sp.diff(sp.sign(r), r), 0)
        lambdified_jacobian_cylinder  = autowrap(lambdified_jacobian_cylinder , backend="f2py", args = [d, phi, theta, alpha, mix, miy, miz, r])


        return lambdified_jacobian_cylinder

    def compute_jacobian_cylinder_loop(self, lambdified_jacobian_cylinder, input_vector):
        result = lambdified_jacobian_cylinder(*input_vector).reshape((5,)).astype('float32')
        return result

    def surface_estimation(self, xyz, clusterlabels, neighbors, surface_class, surface_class_eig):

        mesh_sphere = []
        center_sphere = []
        mesh_cylinder = []
        center_cylinder = []
        T_vis = []
        vector_a = []
        models_fit = []
        error_score_sphere = []
        error_score_cylinder = []
        model_parameter = []


        # pcd.points = o3d.utility.Vector3dVector(xyz)

        # o3d.io.write_point_cloud("camera/pcddata_cylinder.pcd", pcd, print_progress = True)

        # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))
        #
        # pcd_xyz = np.asarray(pcd.points)
        # pcd_normals = np.asarray(pcd.normals)

        # pcd = o3d.io.read_point_cloud("camera/pcddata_cylinder.pcd")
        #
        # xyz     = np.asarray(pcd.points)


        # np.savetxt('camera/pcddata', xyz)
        # np.savetxt('camera/clusterlabels', clusterlabels)

        jac_input_vector = []
        clusters = []
        cluster_idx = []
        cov_mat = []
        d = []
        first_run = True
        cov_mat = []
        cluster_normals = []
        cluster_normals_app = []
        clusters_radius = []
        radius_0 = []
        center_0 = []
        xstart_0 = []
        input_vector = []
        temp = []
        start_timer = 0

        pcd_seg = o3d.geometry.PointCloud()



        ### loop to look at each cluter 1 by 1
        for i in range(np.amax(clusterlabels)+1):

            #print(i,'i')

            #get the index of the cluster that is being classified
            cluster_idx.append(np.where((clusterlabels == i)))

            ## empty arrat for clusters
            clusters = []

            ## extract cluster from pcd
            clusters = np.array(xyz[cluster_idx[i]])

            # pcd_seg.points = o3d.utility.Vector3dVector(clusters)
            #
            # o3d.visualization.draw_geometries([pcd_seg])

            clusters_radius.append(clusters)

            ## tree qurry over the current cluster
            tree = KDTree(np.squeeze(clusters))
            dist, ind = tree.query(np.squeeze(clusters), k = neighbors)

            clusters_ind = np.squeeze(clusters)[ind]

            ## multi processing of normals for current cluster
            cluster_normals = np.asarray(p.map(seg.normals,clusters_ind))

            cluster_normals_app.append(cluster_normals)

            # save cluster_normals
            # np.savetxt('camera/cluster_normals', cluster_normals)

            # #print(clusters,'clusters')

            # #print(len(pcd_normals),'pcd_normals')
            #print(len(clusters),'clusters')

            # cluster_normals = pcd_normals

            #####################################################################################################################
            ################################################## plane ###########################################################
            #####################################################################################################################
            # cluster_mean = (np.mean(clusters,axis = 1))
            #
            # #print(cluster_mean,'cluster_mean')
            #
            # # #print(len(clusters[0]),'clusters len')
            #
            # cov_mat = (1/len(clusters)) * \
            #                (np.dot(np.transpose( np.squeeze(clusters) - cluster_mean),
            #                         ( np.squeeze(clusters) - cluster_mean)))
            # #print(np.shape(clusters),'np.shape(clusters)')

            # input()
            # w,v = LA.eig(cov_mat)
            #
            # #sort normals
            # idx = w.argsort()
            #
            # v = v[:,idx]
            #
            # #keep the smalles normals
            # plane_normal = v[:,0]
            #
            # # #print(plane_normal,'plane_normal')
            # # #print(cov_mat,'cov_mat')
            #
            # ## calculate normal centroid
            # ni_plane = np.mean(clusters,axis = 0)
            # ni_plane = ni_plane/np.linalg.norm(ni_plane)
            #
            # ## initialize z axis
            # z_hat = np.array([0,0,-1])
            #
            # #compute the alignment transofrmation with the z axis
            # r_theta = np.arccos(np.dot(ni_plane,z_hat))
            # r_A = np.cross(ni_plane,z_hat)
            #
            # ## Rodrigues formulation for rmat
            # skew_w = np.array([[0,-r_A[2],r_A[1]],[r_A[2],0,-r_A[0]],[-r_A[1],r_A[0],0]])
            #
            # r_mat = np.eye(3) + skew_w + (skew_w.dot(skew_w))/(1+np.cos(r_theta))
            #
            # ## normals of cluster aligned with z axis
            # nj_alligned = np.dot(np.squeeze(r_mat),np.transpose(clusters))
            #
            # nja = np.mean(nj_alligned,axis = 1)
            # nja = nja/np.linalg.norm(nja)
            #
            # plane_points = nja
            #
            #
            # #d = distance to origin calculated from the mean position
            # distance_to_origin_of_plane = ( -np.transpose(plane_normal) * cluster_mean)
            #
            # #print(distance_to_origin_of_plane,'distance_to_origin_of_plane')
            #
            # cluster_plane_size_max = np.squeeze(np.argmax(clusters, axis=1))
            # cluster_plane_size_min = np.squeeze(np.argmin(clusters, axis=1))
            #
            # #print(cluster_plane_size_max,'cluster_plane_size_max')
            # #print(cluster_plane_size_min,'cluster_plane_size_min')
            # #print(len(np.squeeze(clusters)),'clusters')
            # #print(np.squeeze(clusters)[cluster_plane_size_max[0]][0],'clusters[cluster_plane_size_max[0]]')
            #
            # cluster_plane_size     = np.array([\
            #                                    [np.squeeze(clusters)[cluster_plane_size_max[0]][0] - np.squeeze(clusters)[cluster_plane_size_min[0]][0]],\
            #                                    [np.squeeze(clusters)[cluster_plane_size_max[1]][1] - np.squeeze(clusters)[cluster_plane_size_min[0]][0]],\
            #                                    [np.squeeze(clusters)[cluster_plane_size_max[2]][2] - np.squeeze(clusters)[cluster_plane_size_min[0]][0]]\
            #                                   ])
            #
            # #print(cluster_plane_size,'cluster_plane_size')
            #
            # mesh = o3d.geometry.TriangleMesh.create_box(width=cluster_plane_size[0], height=cluster_plane_size[1], depth=cluster_plane_size[2])
            # mesh = o3d.geometry.TriangleMesh.create_box(width=0.13, height=0.13, depth=0.045)
            #
            #
            # center = np.squeeze(distance_to_origin_of_plane)

            ##print(d,'d')

            #####################################################################################################################
            ################################################## sphere ###########################################################
            #####################################################################################################################

            ### make surface_class, surface_class_eig decide
            ################ classes ####################
            ## table of how to understand surface_class information and surface_class_eig
            # plane   = 0
            # ridge   = 1
            # peak    = 2
            # valley  = 3
            # pit     = 4
            # saddle  = 5
            # if pit / peak fit cylinder
            # if ridge / valley fit sphere


            #print(surface_class_eig[i],'surface_class_eig[i] == 3 or surface_class_eig[i] == 3:')
            #print(surface_class,'surface_class')
            #print(surface_class_eig,'surface_class_eig')


            # cluster_normals = cluster_normals_app
            #
            # #print(np.shape(cluster_normals[0]),'cluster_normals')
            # #print(np.shape(cluster_normals[1]),'cluster_normals')
            # #print(np.shape(np.squeeze(clusters_radius[0])),'clusters')
            #
            # clusters = clusters_radius

            start_timer = time.perf_counter()

            r0 = -(np.dot(len(cluster_normals), np.sum(np.dot(np.transpose(np.squeeze(clusters)) , cluster_normals))) \
                  - np.dot((np.sum(np.transpose(clusters))),np.sum(cluster_normals))) \
                    /((len(cluster_normals)*(np.sum(np.dot(np.transpose(cluster_normals),cluster_normals)))) \
                    -(np.dot(np.sum(np.transpose(cluster_normals)),np.sum(cluster_normals))))

            radius_0.append(r0)

            #print(r0,'r0')
            #print(radius_0,'radius_0')


            # #print(np.sum(cluster_normals,axis = 0),'sumclsuters')
            # #print(np.sum(clusters,axis = 1),'sumclsuters')
            # #print(np.sum(np.dot(r0,cluster_normals)+(cluster_normals),axis = 0),'dot')

            # #print(1/len(cluster_normals),'(1/len(cluster_normals)')


            c0 = np.squeeze((1/len(cluster_normals))*(np.sum(np.dot(r0,cluster_normals)+(clusters),axis = 0)))

            center_0.append(c0)

            #print(c0,'c0')
            #print(center_0,'center_0')

            x0 = np.array([c0[0], c0[1], c0[2], LA.norm(r0)])

            xstart_0.append(x0)

            #print(x0,'x0')
            #print(xstart_0,'xstart_0')

            # decision_variabels =
            length_normals = len(cluster_normals)

            jacobian_sphere = []

            # cxc = c0[0]
            # cyc = c0[1]
            # czc = c0[2]
            # rc = LA.norm(r0)

            # #print((jac_input_vector),'jac_input_vector')
            #

            # #print(jac_input_vector,'jac_input_vector')
            # #print(temp,'temp')
            # #print((jac_input_vector),'jac_input_vector')
            #print(np.shape((jac_input_vector)),'jac_input_vector')

            # np.savetxt('camera/jac_input_vector', jac_input_vector, fmt='%f')

            mix = np.squeeze(np.transpose(clusters)[0,:])
            miy = np.squeeze(np.transpose(clusters)[1,:])
            miz = np.squeeze(np.transpose(clusters)[2,:])


            startf0 = 0

            startf0 = np.sqrt((mix - c0[0])**2 + (miy - c0[1])**2  + (miz - c0[2])**2) - r0
            #print(np.shape(startf0),'startf0')

            # #print(np.shape(temp[0]),'temp')
            # #print(np.shape(temp),'temp')
            clusters = np.squeeze(np.asarray(clusters))
            no_pts = clusters.shape[0]
            initial_x = np.array([[c0[0]],[c0[1]] ,[c0[2]] ,[r0]])

            # #print(np.shape(clusters),'clusters')
            # #print(np.shape(no_pts),'no_pts')
            # #print(np.transpose(initial_x),'initial_x')

            jac_input_vector = np.append(clusters,np.ones((no_pts, 1)).dot(np.transpose(initial_x)),axis = 1)

            # #print(jac_input_vector,'jac_input_vector')

            lambdified_jacobian = seg.lambdifing_jacobian_sphere()

            jacobian_sphere_forloop = []

            # start_timer = time.perf_counter()

            for m in range(len(jac_input_vector)):
                jacobian_sphere_forloop.append(seg.compute_jacobian_sphere_loop( lambdified_jacobian, jac_input_vector[m, :]) )
            jacobian_sphere = np.array(jacobian_sphere_forloop)


            #print('done jacobian_sphere')
            # #print(jacobian_sphere,'jacobian_sphere')
            #print(np.shape(jacobian_sphere),'jacobian_sphere')
            #print(np.shape(jacobian_sphere[0]),'jacobian_sphere len')
            #print(length_normals),'length_normals'
            #print('done')

            # delta = -(dot(Jt*J)^-1*Jt*f0)

            lm_delta = -np.dot( inv(np.dot(np.transpose(jacobian_sphere),jacobian_sphere)), (np.dot(np.transpose(jacobian_sphere),startf0)))

            eps = 10**(-7)

            lamda = 10000
            c = 2

            count = 0
            first_loop = True

            #print(LA.norm(lm_delta),'LA.norm(lm_delta)')
            #print(eps,'eps')

            #print(np.array([c0[0],c0[1] ,c0[2] ,r0]))

            countapp = []
            lm_F_x0_app = []
            jacobian_sphere_forloop = []

            cx  = c0[0]
            cy  = c0[1]
            cz  = c0[2]
            r   = r0


            x_current = np.array([cx ,cy ,cz ,r ])

            #print(x_current,'x_current qwer')

            clusters = np.squeeze(np.asarray(clusters))
            no_pts = clusters.shape[0]
            initial_x = np.array([[cx],[cy],[cz],[r]])

            jac_input_vector = np.append(clusters,np.ones((no_pts, 1)).dot(np.transpose(initial_x)),axis = 1)

            lambdified_jacobian = seg.lambdifing_jacobian_sphere()

            jacobian_sphere_forloop = []
            for m in range(len(jac_input_vector)):
                jacobian_sphere_forloop.append(seg.compute_jacobian_sphere_loop( lambdified_jacobian, jac_input_vector[m, :]) )
            lm_jacobian_sphere = np.array(jacobian_sphere_forloop)

            # H  = dot(Jt,J)
            lm_hessian = np.dot(np.transpose(lm_jacobian_sphere),lm_jacobian_sphere)


            # f0 error
            lm_f0 = np.sqrt((mix - cx)**2 + (miy - cy)**2  + (miz - cz)**2) - abs(r)


            #print(shape(lm_f0),'shape(lm_f0)')
            #print(shape(lm_jacobian_sphere),'shape(lm_jacobian_sphere)')
            #print(shape(lm_delta),'shape(lm_delta)')
            #print(shape(lm_hessian),'shape(lm_hessian)')


            #quadratic loss first round
            lm_F_x0 = np.dot(np.transpose(lm_f0),lm_f0)\
                      + 2 * np.dot(np.dot(np.transpose(lm_f0),lm_jacobian_sphere), lm_delta)\
                      + np.dot(np.transpose(lm_delta),np.dot(lm_hessian,lm_delta))

            for q in range(100):

                count += 1

                lm_hessian_lamda_I = np.identity(4) * lamda

                #print(np.shape(lm_jacobian_sphere),'lm_jacobian_sphere')
                #print(np.shape(lm_f0),'lm_f0')

                lm_delta = -np.dot(inv(lm_hessian + lm_hessian_lamda_I), (np.dot(np.transpose(lm_jacobian_sphere),lm_f0)))


                #solution temp
                delta_x0 = np.array([cx + lm_delta[0], cy + lm_delta[1], cz + lm_delta[2], abs(r) + lm_delta[3]])

                #error temp
                lm_f0_delta = np.sqrt((mix - delta_x0[0])**2 + (miy - delta_x0[1])**2  + (miz - delta_x0[2])**2) - delta_x0[3]

                #quadratlm_F_x0ic loss temp
                lm_F_x0_delta = np.sum(lm_f0_delta * lm_f0_delta)/x_current.shape[0]

                #F error
                # lm_F_x0_delta = np.dot(np.transpose(lm_f0_delta),lm_f0_delta) + 2 * np.dot(np.dot(np.transpose(lm_f0_delta),lm_jacobian_sphere)  , delta_x0) + np.dot(np.transpose(delta_x0),np.dot(lm_hessian,delta_x0))

                countapp.append(count)
                lm_F_x0_app.append(lm_F_x0)

                if(LA.norm(lm_delta) <= eps ):
                    break

                #print('#############################################################')
                #print(count,'count')

                # #print(lm_f0[0],'lm_f0 [0]')
                # #print(lm_f0[1],'lm_f0 [1]')
                #print(lm_F_x0,'lm_F_x0')
                # #print(lm_f0_delta,'lm_f0_delta')

                #print(lm_F_x0_delta,'lm_F_x0_delta2')

                # #print(delta_x0,'delta_x0')
                # #print(lm_delta,'lm_delta')
                # # f0
                # lm_f0 = np.sqrt((mix - delta_x0[0])**2 + (miy - delta_x0[1])**2  + (miz - delta_x0[2])**2) - delta_x0[3]
                #
                # lm_F_x0_delta = np.dot(np.transpose(lm_f0),lm_f0) + 2 * np.dot(np.dot(np.transpose(lm_f0),lm_jacobian_sphere) , lm_f0_delta) + np.dot(np.transpose(lm_delta),np.dot(lm_hessian,lm_delta))


                #print(x_current,'x_current')

                if ( lm_F_x0_delta < lm_F_x0):

                    lamda = lamda/c

                    x_current = x_current + lm_delta

                    cx  = x_current[0]
                    cy  = x_current[1]
                    cz  = x_current[2]
                    r   = x_current[3]


                    #quadratic loss update
                    lm_F_x0 = lm_F_x0_delta

                    #error update
                    lm_f0 = lm_f0_delta

                    clusters = np.squeeze(np.asarray(clusters))
                    no_pts = clusters.shape[0]
                    initial_x = np.array([[cx],[cy],[cz],[r]])

                    jac_input_vector = np.append(clusters,np.ones((no_pts, 1)).dot(np.transpose(initial_x)),axis = 1)

                    jacobian_sphere_forloop = []
                    for m in range(len(jac_input_vector)):
                        jacobian_sphere_forloop.append(seg.compute_jacobian_sphere_loop( lambdified_jacobian, jac_input_vector[m, :]) )
                    lm_jacobian_sphere = np.array(jacobian_sphere_forloop)

                    # H  = dot(Jt,J)
                    lm_hessian = np.dot(np.transpose(lm_jacobian_sphere),lm_jacobian_sphere)




                else:
                    lamda = c * lamda

                #print(lamda,'lamda')
                #print(c,'c')
                #print(x_current,'x_current')
                #print(LA.norm(lm_delta),'LA.norm(lm_delta)')

            cx_sphere  = x_current[0]
            cy_sphere  = x_current[1]
            cz_sphere  = x_current[2]
            r_sphere   = x_current[3]

            end_timer = time.perf_counter()
            final_time = end_timer - start_timer
            #print(final_time,'final_time')





            #### sphere construction for visualization ####
            center_sphere = [cx_sphere,  cy_sphere , cz_sphere ]

            radius_sphere = abs(r_sphere)

            mesh_sphere = (o3d.geometry.TriangleMesh.create_sphere( radius_sphere ))
            world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.07, origin=[0, 0, 0])

            mesh_sphere.paint_uniform_color([0.51, 0, 0.14])

            mesh_sphere_app = mesh_sphere.translate((center_sphere[0],center_sphere[1],center_sphere[2]))
            world_frame_app = world_frame.translate((center_sphere[0],center_sphere[1],center_sphere[2]))


            # models_fit.append(mesh_sphere_app)
            # models_fit.append(world_frame_app)

        #####################################################################################################################
        ################################################## cylinder ###########################################################
        #####################################################################################################################


            #print(clusters,'clusters')
            #print(np.shape(clusters),'clusters')
            #print(cluster_normals,'cluster_normals')
            #print(np.shape(cluster_normals),'cluster_normals')


            no_pts = clusters.shape[0]

            ni = np.mean(cluster_normals,axis = 0)

            n_bar = ni/np.linalg.norm(ni)

            ## initialize z axis
            z_hat = np.array([0,0,1])

            r_theta = np.arccos(np.dot(n_bar,z_hat))
            r_A = np.cross(n_bar,z_hat)
            #print(r_theta,'r_theta')
            ## Rodrigues formula`tion for rmat

            skew_w = np.array([[0,-r_A[2],r_A[1]],[r_A[2],0,-r_A[0]],[-r_A[1],r_A[0],0]])

            r_mat = np.eye(3) + skew_w + (skew_w.dot(skew_w))/(1+np.cos(r_theta))

            ni_apo = np.transpose(np.dot(np.squeeze(r_mat),np.transpose(cluster_normals)))

            # cov_mat = (1/len(ni_apo.shape[0])) * \
            #                (np.dot( np.transpose( np.squeeze(ni_apo) - (ni_apo_mean)), ( np.squeeze(ni_apo) - (ni_apo_mean))))

            ni_apo = ni_apo[:, :2]

            #print(ni_apo,'ni_apo')
            #print(shape(ni_apo),'ni_apo')

            cov_mat = (1/ni_apo.shape[0]) * (ni_apo.T).dot(ni_apo)

            #print(cov_mat,'cov_mat')

            w,v = LA.eig(cov_mat)

            #sort normals
            idx = w.argsort()[::-1]

            #print(w,'w')
            #print(idx,'idx')
            v = v[:,idx]

            #print(v,'v')

            #keep the smalles normals
            amin = np.array([v[1,0],v[1,1], 0 ])

            #print(amin,'amin')

            a0 = np.dot(inv(r_mat),amin)

            #print(a0,'a0')

            ## above should be fine in accordance with help code
            ## a0 looks ok

            if a0[2] >  0:
                z_hat = np.array([0,0,-1])
            else:
                z_hat = np  .array([0,0,1])

            #print(z_hat,'z_hat')

            A0_theta = np.arccos(np.dot(a0, z_hat))
            A0_A = np.cross(a0,z_hat)

            ## Rodrigues formula`tion for rmat

            skew_w = np.array([[0,-A0_A[2],A0_A[1]],[A0_A[2],0,-A0_A[0]],[-A0_A[1],A0_A[0],0]])

            r_mat_a0 = np.eye(3) + skew_w + (skew_w.dot(skew_w))/(1+np.cos(A0_theta))


            reorient_cluster_normal = np.transpose(np.dot(r_mat_a0,np.transpose(cluster_normals)))
            reorient_cluster_normal[:,2] = 0
            #print(shape(reorient_cluster_normal),'reorient_cluster_normal')

            reorient_cluster_xyz = np.transpose(np.dot(r_mat_a0, clusters.transpose()))
            reorient_cluster_xyz[:,2]    = 0

            #print(shape(reorient_cluster_normal),'reorient_cluster_normal')
            #print(shape(reorient_cluster_xyz),'reorient_cluster_xyz')

            r0 = -(np.dot(len(reorient_cluster_xyz), np.sum(np.dot(np.transpose(np.squeeze(reorient_cluster_normal)) , reorient_cluster_xyz))) \
                  - np.dot((np.sum(np.transpose(reorient_cluster_normal))) , np.sum(reorient_cluster_xyz))) \
                    /((len(reorient_cluster_xyz) * (np.sum(np.dot(np.transpose(reorient_cluster_normal),reorient_cluster_normal)))) \
                    -(np.dot(np.sum(np.transpose(reorient_cluster_normal)),np.sum(reorient_cluster_normal))))

            #print(r0,'r0')

            # t1 = np.sum(reorient_cluster_xyz * reorient_cluster_normal)
            # t2 = np.sum(np.sum(reorient_cluster_xyz, axis=0) * np.sum(reorient_cluster_normal, axis=0))
            # t3 = np.sum(reorient_cluster_normal * reorient_cluster_normal)
            # t4 = np.sum(np.sum(reorient_cluster_normal, axis=0) * np.sum(reorient_cluster_normal, axis=0))
            #
            # r0 = -(no_pts*t1 - t2)/(no_pts*t3 - t4)


            #print(shape(reorient_cluster_normal),'reorient_cluster_normal')
            #print(shape(reorient_cluster_xyz),'reorient_cluster_xyz')
            #print((reorient_cluster_normal[0]),'reorient_cluster_normal')
            #print((reorient_cluster_xyz[0]),'reorient_cluster_xyz')

            c0_apo = np.squeeze((1/len(reorient_cluster_xyz))\
                        *(np.sum(r0*reorient_cluster_normal+(reorient_cluster_xyz),axis = 0)))


            #print(c0_apo,'c0_apo')
            #print(r_mat,'r_mat')
            #print(r_mat_a0,'r_mat_a0')

            c0_apo = np.squeeze(c0_apo)
            a0 = np.squeeze(a0)

            c0 = np.dot(inv(r_mat_a0),c0_apo)


            #print(a0,'a0')
            #print(c0,'c0')
            #print(r0,'r0')



            p0 = c0 - np.dot(a0,np.dot(np.transpose(c0),a0))

            #print(p0,'p0')

            d0 = np.linalg.norm(p0)

            n_polar = p0/d0

            #print(n_polar,'n_polar')

            n_polar_theta = np.arctan2(np.sqrt(n_polar[0]*n_polar[0] + n_polar[1]*n_polar[1]), n_polar[2])

            n_polar_phi = np.arctan2(n_polar[1],n_polar[0])


            #print(n_polar_theta,'n_polar_theta')
            #print(n_polar_phi,'n_polar_phi')
            #print(d0 ,'d0 ')

            n_theta = np.transpose(np.array([ np.cos(n_polar_phi)*np.cos(n_polar_theta) , np.sin(n_polar_phi)*np.cos(n_polar_theta) , -np.sin(n_polar_theta) ]))

            n_phi = np.array([-np.sin( n_polar_phi ), np.cos(n_polar_phi) , 0 ])

            cos_alpha = (float((a0[0]      * n_phi[1] - a0[1]      * n_phi[0])/\
                                (n_theta[0] * n_phi[1] - n_theta[1] * n_phi[0]  )))

            sin_alpha = (float((a0[0]      * n_theta[1] - a0[1]      * n_theta[0])/\
                                (n_phi[0]   * n_theta[1] - n_phi[1]   * n_theta[0])))


            alpha0 = np.arctan2(sin_alpha, cos_alpha)


            #print(cos_alpha,'cos_alpha')
            #print(sin_alpha,'sin_alpha')
            #print(alpha0,'alpha0')


            # cos_alpha = a0[2]/n_theta[2]
            #
            # sin_alpha = (a0[0] - n_theta[0]*a0[2]/n_theta[2])/n_phi[0]
            #
            # #print(cos_alpha,'cos_alpha')
            # #print(sin_alpha,'sin_alpha')


            alpha0 = np.arctan2(sin_alpha, cos_alpha)


            #print(d0,'d0')
            #print(n_polar_theta,'n_polar_theta')
            #print(n_polar_phi,'n_polar_phi')
            #print(alpha0,'alpha0')
            #print(r0,'r0')

            initial_est = np.array([ [d0], [n_polar_theta] , [n_polar_phi] , [alpha0] ,[r0] ])

            # #print(type(alpha0),'type(alpha0)')
            # #print(type(cos_alpha),'type(cos_alpha)')
            # #print(type(sin_alpha),'type(sin_alpha)')

            pcd.points = o3d.utility.Vector3dVector(xyz)
            #print(f'Center of pcd: {pcd.get_center()}')



            vector_n = np.array([[np.cos(n_polar_phi) * np.sin(n_polar_theta) ], \
                               [np.sin(n_polar_phi) * np.sin(n_polar_theta) ], \
                               [np.cos(n_polar_theta)                    ]])

            vector_n_theta = np.array([[np.cos(n_polar_phi) * np.cos(n_polar_theta) ], \
                                     [np.sin(n_polar_phi) * np.cos(n_polar_theta) ], \
                                     [-np.sin(n_polar_theta)                   ]])

            vector_n_phi = np.array([[-np.sin(n_polar_phi)   ], \
                                  [np.cos(n_polar_phi)    ], \
                                  [0                   ]])

            vector_a = np.squeeze(( vector_n_theta * np.cos(alpha0) + vector_n_phi * np.sin(alpha0) ))

            vector_a = np.array([[vector_a[0]],[vector_a[1]] ,[vector_a[2]] ])

            #print(vector_a,'vector_a')


            mix = np.squeeze(np.transpose(clusters)[0,:])
            miy = np.squeeze(np.transpose(clusters)[1,:])
            miz = np.squeeze(np.transpose(clusters)[2,:])

            vector_mi = np.transpose(np.array([ mix, miy, miz ]))

            vector_p = np.squeeze(np.transpose(d0*vector_n ))

            vector_p = np.array([[vector_p[0]],[vector_p[1]] ,[vector_p[2]] ])

            ##print(type(vector_p + vector_a * np.transpose(np.dot(vector_mi , vector_p)) * vector_a - np.transpose(vector_mi)),'lm_f_x0_cylinder1')
            ##print(np.linalg.norm(np.squeeze(vector_p + vector_a * np.transpose(np.dot(vector_mi , vector_p)) * vector_a - np.transpose(vector_mi))[0]),'np.linalg.norm(vector_p + vector_a * np.transpose(np.dot(vector_mi , vector_p)) * vector_a - np.transpose(vector_mi))')

            cylinder_D = np.transpose(vector_p + vector_a * np.transpose(vector_mi - np.transpose(vector_p)) * vector_a - np.transpose(vector_mi))

            #error
            lm_f_x0_cylinder = (np.linalg.norm(cylinder_D, axis = 1) - abs(r0)).reshape((no_pts, 1))

            #lost
            lm_F_x0_cylinder = np.sum( lm_f_x0_cylinder * lm_f_x0_cylinder )/ clusters.shape[0]

            #print(lm_F_x0_cylinder,'lm_F_x0_cylinder')



            vector_a = np.squeeze(np.transpose(vector_a))
            vector_p = np.squeeze(np.transpose(vector_p))

            #print(np.shape(clusters),'clusters')
            #print(np.shape(vector_p),'vector_p')
            #print((vector_p),'vector_p')
            #print(np.shape(vector_a),'vector_a')
            #print((vector_a),'vector_a')

            temp                 = np.sum(((clusters) - vector_p)*vector_a, axis = 1).reshape((no_pts, 1))
            D                    = vector_p + temp.dot(vector_a.reshape((1, 3))) - clusters
            signed_distances     = (np.linalg.norm(D, axis = 1) - abs(r0)).reshape((no_pts, 1))
            # #print(signed_distances,'signed_distances')

            lm_F_x0_cylinder                 = np.sum(signed_distances*signed_distances) / no_pts

            #print(lm_F_x0_cylinder,'lm_F_x0_cylinder')

            lm_f0_cylinder = lm_f_x0_cylinder

            lm_f0_cylinder = signed_distances



            jac_input_vector_cylinder = np.append(clusters,np.ones((no_pts, 1)).dot(np.transpose(initial_est)),axis = 1)

            lambdifing_jacobian_cylinder  = seg.lambdifing_jacobian_cylinder()

            lm_jacobian_cylinder_forloop = []
            for m in range(len(jac_input_vector_cylinder)):
                lm_jacobian_cylinder_forloop.append(seg.compute_jacobian_cylinder_loop( lambdifing_jacobian_cylinder, jac_input_vector_cylinder[m, :]) )
            lm_jacobian_cylinder = np.array(lm_jacobian_cylinder_forloop)

            # #print(jacobian_cylinder,'jacobian_cylinder')

            lm_delta_cylinder = -np.dot( inv(np.dot(np.transpose(lm_jacobian_cylinder),lm_jacobian_cylinder)), (np.dot(np.transpose(lm_jacobian_cylinder),lm_f_x0_cylinder)))

            eps = 10**(-7)

            lamda = 10000
            c = 2

            count_cylinder = 0

            countapp_cylinder = []
            lm_F_x0_app_cylinder = []
            # jacobian_cylinder_forloop = []

            x_current = initial_est

            #print(x_current,'x_current')

            d = d0
            theta = n_polar_theta
            phi = n_polar_phi
            alpha = alpha0
            r = r0

            #print(d0,'d0')
            #print(theta,'theta')
            #print(phi,'phi')
            #print(alpha,'alpha')
            #print(r,'r')

            lm_hessian_cylinder = np.dot(np.transpose(lm_jacobian_cylinder),lm_jacobian_cylinder)

            for q in range(100):

                count_cylinder += 1

                #print(count,'count')
                #print(lamda,'lamda')

                lm_hessian_lamda_I = np.identity(5) * lamda


                # lm_delta_cylinder = -np.dot( inv(np.dot(np.transpose(lm_jacobian_cylinder),lm_jacobian_cylinder)), (np.dot(np.transpose(lm_jacobian_cylinder),startf0)))
                lm_delta_cylinder = -np.dot(inv(lm_hessian_cylinder + lm_hessian_lamda_I), (np.dot(np.transpose(lm_jacobian_cylinder),lm_f0_cylinder)))

                #solution temp
                delta_x = np.array([d + lm_delta_cylinder[0], theta + lm_delta_cylinder[1], phi + lm_delta_cylinder[2],\
                                    alpha + lm_delta_cylinder[3], abs(r) + lm_delta_cylinder[4]])

                #print(lm_delta_cylinder,'lm_delta_cylinder')


                # error lm_f_x0_cylinder delta
                vector_n = np.array([[np.cos(delta_x[2]) * np.sin(delta_x[1]) ], \
                                   [np.sin(delta_x[2]) * np.sin(delta_x[1]) ], \
                                   [np.cos(delta_x[1])                    ]])

                vector_n_theta = np.array([[np.cos(delta_x[2]) * np.cos(delta_x[1]) ], \
                                         [np.sin(delta_x[2]) * np.cos(delta_x[1]) ], \
                                         [-np.sin(delta_x[1])                   ]])

                vector_n_phi = np.array([[-np.sin(delta_x[2])], \
                                         [np.cos(delta_x[2]) ], \
                                         [0                   ]])

                vector_n_phi = np.array([vector_n_phi[0][0], vector_n_phi[1][0] , vector_n_phi[2]])

                vector_a = np.squeeze(( vector_n_theta * np.cos(delta_x[3]) + vector_n_phi * np.sin(delta_x[3]) ))

                vector_a = (np.array([[vector_a[0][0]],[vector_a[1][1]] ,[vector_a[2][2]] ]))


                mix = np.squeeze(np.transpose(clusters)[0,:])
                miy = np.squeeze(np.transpose(clusters)[1,:])
                miz = np.squeeze(np.transpose(clusters)[2,:])

                vector_mi = np.transpose(np.array([ mix, miy, miz ]))

                vector_p = np.squeeze(np.transpose(delta_x[0]*vector_n ))

                vector_p = np.array([[vector_p[0]],[vector_p[1]] ,[vector_p[2]] ])


                # cylinder_D = np.transpose(vector_p + vector_a\
                #              * np.transpose(vector_mi - np.transpose(vector_p)) * vector_a\
                #               - np.transpose(vector_mi))

                # #error
                # lm_f0_delta_cylinder = (np.linalg.norm(cylinder_D, axis = 1) - abs(delta_x[4])).reshape((no_pts, 1))
                #
                # #quadratlm_F_x0ic loss temp
                # lm_F_x0_delta_cylinder = np.sum(lm_f0_delta_cylinder * lm_f0_delta_cylinder)/no_pts


                vector_a = np.squeeze(np.transpose(vector_a))
                vector_p = np.squeeze(np.transpose(vector_p))

                # #print(np.shape(clusters),'clusters')
                # #print(np.shape(vector_p),'vector_p')
                # #print((vector_p),'vector_p')
                # #print(np.shape(vector_a),'vector_a')
                # #print((vector_a),'vector_a')

                temp                 = np.sum(((clusters) - vector_p)*vector_a, axis = 1).reshape((no_pts, 1))
                D                    = vector_p + temp.dot(vector_a.reshape((1, 3))) - clusters
                signed_distances     = (np.linalg.norm(D, axis = 1) - abs(r)).reshape((no_pts, 1))
                # #print(signed_distances,'signed_distances')

                lm_F_x0_delta_cylinder                 = np.sum(signed_distances*signed_distances) / no_pts

                # #print(lm_F_x0_delta_cylinder,'lm_F_x0_delta_cylinder')

                countapp_cylinder.append(count_cylinder)
                lm_F_x0_app_cylinder.append(lm_F_x0_cylinder)

                if(LA.norm(lm_delta_cylinder) <= eps ):
                    break

                #print(lm_F_x0_delta_cylinder,'lm_F_x0_delta_cylinder')
                #print(lm_F_x0_cylinder,'lm_F_x0_cylinder')
                #print(x_current,'x_current')

                if ( lm_F_x0_delta_cylinder < lm_F_x0_cylinder):

                    lamda = lamda/c

                    x_current = x_current + lm_delta_cylinder

                    d  = x_current[0]
                    theta  = x_current[1]
                    phi  = x_current[2]
                    alpha  = x_current[3]
                    r  = x_current[4]


                    #quadratic loss update
                    lm_F_x0_cylinder = lm_F_x0_delta_cylinder

                    #error update
                    lm_f0_cylinder = signed_distances

                    clusters = np.squeeze(np.asarray(clusters))

                    # x_current = np.array([ [d], [theta] , [phi] , [alpha] ,[r] ])

                    jac_input_vector_cylinder = np.append(clusters,np.ones((no_pts, 1)).dot(np.transpose(x_current)),axis = 1)

                    # lambdifing_jacobian_cylinder  = seg.lambdifing_jacobian_cylinder ()

                    lm_jacobian_cylinder_forloop = []
                    for m in range(len(jac_input_vector_cylinder)):
                        lm_jacobian_cylinder_forloop.append(seg.compute_jacobian_cylinder_loop( lambdifing_jacobian_cylinder, jac_input_vector_cylinder[m, :]) )
                    lm_jacobian_cylinder = np.array(lm_jacobian_cylinder_forloop)

                    lm_hessian_cylinder = np.dot(np.transpose(lm_jacobian_cylinder),lm_jacobian_cylinder)

                else:
                    lamda = c * lamda

            ################################### after loop #####################################

            # x_current = np.array([[0.50015691], [0.6259656], [0.82009577], [0.22393045], [0.02720507 ]])

            # error lm_f_x0_cylinder delta
            vector_n = np.array([[np.cos(x_current[2]) * np.sin(x_current[1]) ], \
                               [np.sin(x_current[2]) * np.sin(x_current[1]) ], \
                               [np.cos(x_current[1])                    ]])

            vector_n_theta = np.array([[np.cos(x_current[2]) * np.cos(x_current[1]) ], \
                                     [np.sin(x_current[2]) * np.cos(x_current[1]) ], \
                                     [-np.sin(x_current[1])                   ]])

            vector_n_phi = np.array([[-np.sin(x_current[2])], \
                                     [np.cos(x_current[2]) ], \
                                     [0                   ]])

            vector_n_phi = np.array([vector_n_phi[0][0], vector_n_phi[1][0] , vector_n_phi[2]])

            vector_a = np.squeeze(( vector_n_theta * np.cos(x_current[3]) + vector_n_phi * np.sin(x_current[3]) ))

            vector_a = (np.array([[vector_a[0][0]],[vector_a[1][1]] ,[vector_a[2][2]] ]))


            mix = np.squeeze(np.transpose(clusters)[0,:])
            miy = np.squeeze(np.transpose(clusters)[1,:])
            miz = np.squeeze(np.transpose(clusters)[2,:])

            vector_mi = np.transpose(np.array([ mix, miy, miz ]))

            vector_p = np.squeeze(np.transpose(x_current[0]*vector_n ))

            vector_p = np.array([[vector_p[0]],[vector_p[1]] ,[vector_p[2]] ])


            # cylinder_D = np.transpose(vector_p + vector_a\
            #              * np.transpose(vector_mi - np.transpose(vector_p)) * vector_a\
            #               - np.transpose(vector_mi))

            # #error
            # lm_f0_delta_cylinder = (np.linalg.norm(cylinder_D, axis = 1) - abs(delta_x[4])).reshape((no_pts, 1))
            #
            # #quadratlm_F_x0ic loss temp
            # lm_F_x0_delta_cylinder = np.sum(lm_f0_delta_cylinder * lm_f0_delta_cylinder)/no_pts


            vector_a = np.squeeze(np.transpose(vector_a))
            vector_p = np.squeeze(np.transpose(vector_p))

            # #print(np.shape(clusters),'clusters')
            # #print(np.shape(vector_p),'vector_p')
            # #print((vector_p),'vector_p')
            # #print(np.shape(vector_a),'vector_a')
            # #print((vector_a),'vector_a')

            temp                 = np.sum(((clusters) - vector_p)*vector_a, axis = 1).reshape((no_pts, 1))
            D                    = vector_p + temp.dot(vector_a.reshape((1, 3)))
            signed_distances     = (np.linalg.norm(D, axis = 1) - abs(r)).reshape((no_pts, 1))
            # #print(signed_distances,'signed_distances')

            lm_F_x0_delta_cylinder                 = np.sum(signed_distances*signed_distances) / no_pts

            #print(D ,'D ')


            min_index = np.argmin(D[:, 0])
            max_index = np.argmax(D[:, 0])

            height_cylinder = np.linalg.norm(D[min_index, :] - D[max_index, :])

            directional_vector   = (D[max_index, :] - D[min_index, :])/np.linalg.norm(D[max_index, :] - D[min_index, :])
            center_cylinder = D[min_index, :]  + directional_vector * height_cylinder/2

            #print(x_current,'x_current')

            radius_cylinder = x_current[4]

            #print(center_cylinder,'center_cylinder')
            #print(height_cylinder,'height_cylinder')





            ############## cylinder rotation
            mesh_cylinder = (o3d.geometry.TriangleMesh.create_cylinder(radius = abs(radius_cylinder), height = height_cylinder, resolution = 100))
            world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.07, origin=[0, 0, 0])

            mesh_cylinder_points = np.asarray((o3d.geometry.TriangleMesh.sample_points_uniformly(mesh_cylinder,100)).points)

            # vector_a = np.array([ 0.38674888,  0.74546022, -0.54287601])

            r_mat_cylinder =  np.dot(np.array([0,0,1]),vector_a)

            A0_theta = np.arccos(np.dot(np.array([0,0,1]), vector_a))
            A0_A = np.cross(np.array([0,0,1]),vector_a)

            #print(vector_a,'vector_a')


            ## Rodrigues formula`tion for rmat
            skew_w = np.array([[0,-A0_A[2],A0_A[1]],[A0_A[2],0,-A0_A[0]],[-A0_A[1],A0_A[0],0]])

            r_mat_cylinder = np.eye(3) + skew_w + (skew_w.dot(skew_w))/(1+np.cos(A0_theta))

            mesh_cylinder.rotate(r_mat_cylinder,[0,0,0])
            world_frame.rotate(r_mat_cylinder,[0,0,0])

            mesh_cylinder.compute_vertex_normals()
            mesh_cylinder.paint_uniform_color([0, 0.51, 0.14])

            mesh_cylinder = mesh_cylinder + world_frame

            mesh_cylinder_app = mesh_cylinder.translate((center_cylinder[0],center_cylinder[1],center_cylinder[2]))
            world_frame_app = world_frame.translate((center_cylinder[0],center_cylinder[1],center_cylinder[2]))

            ###################

            #print(i,'i')
            #print(lm_F_x0_cylinder,'lm_F_x0_cylinder')
            #print(lm_F_x0,'lm_F_x0 sphere')

            # pcd_seg.points = o3d.utility.Vector3dVector(clusters)
            #
            # o3d.visualization.draw_geometries([pcd_seg])

            error_score_sphere.append(lm_F_x0)
            error_score_cylinder.append(lm_F_x0_cylinder)


            if lm_F_x0_cylinder > lm_F_x0:
                #### sphere #####
                plt.figure()

                plt.axis([0, count, 0, np.max(lm_F_x0_app) + 0.01])
                plt.scatter(countapp, lm_F_x0_app)


                plt.savefig("fittingmodel"+str(i))

                models_fit.append(mesh_sphere_app)
                models_fit.append(world_frame_app)

                mesh_sphere_app = 0
                world_frame_app = 0

                model_parameter.append(x_current)

            elif lm_F_x0_cylinder < lm_F_x0:

                plt.figure()

                plt.axis([0, count_cylinder, 0, np.max(lm_F_x0_app_cylinder) + 0.01])
                plt.scatter(countapp_cylinder, lm_F_x0_app_cylinder)

                plt.savefig("fittingmodel"+str(i))

                models_fit.append(mesh_cylinder_app)
                mesh_cylinder_app = 0

                model_parameter.append(x_current)


        # surface_class, surface_class_eig
        with open('results.csv','a') as csvfile:
            np.savetxt(csvfile, error_score_sphere,delimiter=',',header='error sphere ########### round 1 ########### ',fmt='%s', comments='')
            np.savetxt(csvfile,error_score_cylinder,delimiter=',',header='error cylinder',fmt='%s', comments='')
            np.savetxt(csvfile,surface_class,delimiter=',',header='surface_class',fmt='%s', comments='')
            np.savetxt(csvfile,surface_class_eig,delimiter=',',header='surface_class_eig',fmt='%s', comments='')
            for w in range(len(model_parameter)):
                np.savetxt(csvfile,model_parameter[w],delimiter=',',header='model parameters '+str(w),fmt='%s', comments='')


        return models_fit


if __name__ == '__main__':
    cam = Camera()
    seg = Segmentation()
    p = Pool()

    pcd = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()

    vis.create_window("Point Clouds", width=640, height=480)
    added = True

    #varibles for filtering
    neighbors = 30
    radius = 0.02
    minpoints = 30

    #visualization variables
    test_surface_normals = False
    visualize_Filter_Radius = False
    visualize_db_clusters = True
    draw_aruco_corner = True
    draw2d_aruco_marker = False
    read_from_file = True




    while 1:

        # return varibles
        surface_class = []
        surface_class_eig = []

        #compute the rgb and depth img
        rgb_img, depth_img = cam.stream(colored_depth=False)

        #generate pcd
        xyz = cam.generate_pcd(depth_img)

        pcd.points = o3d.utility.Vector3dVector(xyz)

        o3d.visualization.draw_geometries([pcd])

        #remove bin from point cloud
        binxyz = seg.bin_removal(xyz,read_from_file, rgb_img , draw = draw_aruco_corner, draw2d = draw2d_aruco_marker)

        #neighbor filtering of pcd
        fnxyz = seg.filter_neighbors(binxyz, neighbors)

        #radius filtering of pcd
        frxyz = seg.filter_radius(fnxyz, radius, minpoints,visualize_Filter_Radius)

        #surface normals estimation of pcd and alignment
        normals = seg.surface_normal_estimation(frxyz, neighbors, test_surface_normals)

        #Crease_removal of the pcd
        crxyz = seg.crease_removal(frxyz,normals,neighbors)

        #surface normals estimation of pcd and alignment to see if the result is better
        normals = seg.surface_normal_estimation(crxyz, neighbors, test_surface_normals)

        #clustering/segmentate of pcd
        labels = seg.segmentate(crxyz,normals,visualize_db_clusters)

        #### first prediction of shape #####
        surface_class, surface_class_eig = seg.non_parametric_surface_class(crxyz, labels, neighbors)


        models_fit = seg.surface_estimation(crxyz, labels, neighbors, surface_class, surface_class_eig)

        #visualize the pcd
        pcd.points = o3d.utility.Vector3dVector(crxyz)

        #Transfor viewpoint to look form the camera perspective
        # pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        # mesh.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        #visualize the depth_img with color/gray scale
        #rgb_img, depth_img = cam.stream(colored_depth=True)

        ## visualize rgb and depth image
        #cv2.imshow("rgb", rgb_img)
        #cv2.imshow("depth", depth_img)
        #cv2.waitKey(1)

        models_fit.append(pcd)

        #print(models_fit,'models_fit')

        # visualize point cloud caculated from the depth image
        # if added == True:
        #     vis.add_geometry(pcd)
        #     added = False
        # vis.update_geometry(pcd)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        o3d.visualization.draw_geometries(models_fit,point_show_normal=True)
        vis.poll_events()
        vis.update_renderer()
