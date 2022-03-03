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
import functools
from multiprocessing.pool import ThreadPool
from numpy.linalg import inv


def sphere_mini_LM(rc0, clusters):

    # print(rc0,'rc0')
    # print(clusters,'clusters')
    # print(length_normals,'length_normals')
    length_normals = len(clusters)
    c0 = np.array([rc0[0],rc0[1],rc0[2]])
    # print(c0,'c0')
    return (1/length_normals)*np.sum(np.exp2(LA.norm(np.squeeze(clusters) - c0)-rc0[3]))

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
        print(Xi,'xi')
        print(cov,'cov')

        w,v = LA.eig(cov)

        #sort normals
        idx = w.argsort()
        w = w[idx]

        v = v[:,idx]

        #keep the biggest normals
        normals = w[:,1]

        print(w,'w_max')

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

        print(w,'w_min')

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

            temp_index = np.where(Cosine_Similarity[i] > 0.65) ## 0.75 is good

            #keep points that does not span a crease
            if len(temp_index[0]) >  29:
                not_creaseind[i] = int(i)

        #point cloud with creases removed
        xyz = xyz[not_creaseind.astype(int)]

        return xyz

    def segmentate(self, xyz, normals, visualize):

        #doing a dbscan to attemp a clustering of points in the pcd
        db = DBSCAN(eps=0.02, min_samples=50).fit(xyz) # 0.022 & 50
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        #print(labels,'labels')
        labels_unq = np.unique(labels)
        #print(labels_unq,'labels_unq')

        #save the clusters and the noise from the scan
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)


        #visualize the clusters
        if visualize == True:
            fig = plt.figure()
            ax = plt.axes(projection='3d')

            scatter = ax.scatter3D(xyz[:,0],xyz[:,1],xyz[:,2], c=db.labels_, cmap='jet')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            legend1 = ax.legend(scatter.legend_elements(num=len(labels_unq)), loc="upper left", title="Ranking")
            ax.add_artist(legend1)
            plt.show()

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

                    np.savetxt('camera/frame2.txt', cam2markerHom)
        if read_from_file == True:
            cam2markerHom = np.loadtxt('camera/frame2.txt')


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
        # # Best
        bin_width    =  0.34 #M real is 0.34 #0.22
        bin_length   =  0.44 #M real is 0.44 #0.39
        bin_height   =  0.5 #M real is 0.2 #0.175

        condt = np.where((markerpcd[0] >= -0.05)  & (markerpcd[0] <= bin_width) &
                         (markerpcd[1] <= -0.05) & (markerpcd[1] >= -bin_length) &
                         (markerpcd[2] >= 0.2) & (markerpcd[2] <= bin_height) )

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

            print(i,'i')

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

            print(nja,'nja')

            ## decompose the current cluster for better computation
            decomp_cluster = np.transpose(nj_alligned)
            neighbors_decomp = 150

            if (len(decomp_cluster) < neighbors_decomp):
                neighbors_decomp = len(decomp_cluster)

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
            print(eth,'eth')
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
            print(convex_guess,'convex_guess')
            print(concave_guess,'concave_guess')
            print(neither_guess,'neither_guess')
            print(np.mean(mse_max_app),'mse_max_app')
            print(np.mean(mse_min_app),'mse_min_app')

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
            print(unique,'unique')
            print(counts,'counts')
            print(unique_eig,'unique_eig')
            print(counts_eig,'counts_eig')
            print(counts_uni_conxeity,'counts_uni_conxeity')

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

            print(np.squeeze(shape_guess),'shape_guess')

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
            print(np.squeeze(shape_guess),'shape_guess')
            print(np.squeeze(shape_guess_eig),'shape_guess_eig')
            print(np.squeeze(convexity_guess),'convexity_guess')
            print(np.squeeze(shape_guess_procent),'shape_guess_procent')
            print(np.mean(np.squeeze(shape_guess_procent)),'shape_guess_procent_mean')
            print(np.squeeze(convexity_guess_procent),'convexity_guess_procent')
            print((np.mean(np.squeeze(convexity_guess_procent)) + np.mean(np.squeeze(shape_guess_procent)))/2,'mean mean')
            print(surface_class,'surface_class')
            print(surface_class_eig,'surface_class_eig')
        return

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

    def compute_jacobian_sphere(self, input_vector):

        # input_vector = np.loadtxt('camera/jac_input_vector')

        # input_vector = self.jacobian_sphere_input_vector

        self.len = len(input_vector)

        # if self.mix == 0:
        #     print(self.len,'self.len = len(input_vector)')
        #     print(np.shape(input_vector),'input_vector jac sphere func')

        self.mix = input_vector[0]
        self.miy = input_vector[1]
        self.miz = input_vector[2]
        cx  = input_vector[3]
        cy  = input_vector[4]
        cz  = input_vector[5]
        r   = input_vector[6]

        jacobian = self.jacobian_sphere(cx,cy,cz,r)

        return jacobian

    def execute_jacobian(self, input_vector):

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

    def surface_estimation(self, xyz, clusterlabels, neighbors):

        pcd.points = o3d.utility.Vector3dVector(xyz)

        o3d.io.write_point_cloud("camera/pcddata.pcd", pcd)
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

        ### loop to look at each cluter 1 by 1
        for i in range(np.amax(clusterlabels)+1):

            print(i,'i')

            #get the index of the cluster that is being classified
            cluster_idx.append(np.where((clusterlabels == i)))

            ## empty arrat for clusters
            clusters = []

            ## extract cluster from pcd
            clusters.append(xyz[cluster_idx[i]])

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


            cluster_mean = (np.mean(clusters,axis = 1))


            ##### Plane #####

            # print(len(clusters[0]),'clusters len')
            #
            # cov_mat.append( (1/len(clusters)) *
            #                (np.dot(( np.squeeze(clusters) - cluster_mean),
            #                         np.transpose( np.squeeze(clusters) - cluster_mean))))
            #
            # cov_mat_normals = np.asarray(p.map(seg.normals,cov_mat))
            #
            #
            #
            # #d = distance to origin calculated from the mean position
            # d.append( -np.transpose(cov_mat_normals) * cluster_mean)


            #print(d,'d')

            ###### sphere ######

            # cluster_normals = cluster_normals_app
            #
            # print(np.shape(cluster_normals[0]),'cluster_normals')
            # print(np.shape(cluster_normals[1]),'cluster_normals')
            # print(np.shape(np.squeeze(clusters_radius[0])),'clusters')
            #
            # clusters = clusters_radius

            r0 = -(np.dot(len(cluster_normals), np.sum(np.dot(np.transpose(np.squeeze(clusters)) , cluster_normals))) \
                  - np.dot((np.sum(np.transpose(clusters))),np.sum(cluster_normals))) \
                    /((len(cluster_normals)*(np.sum(np.dot(np.transpose(cluster_normals),cluster_normals)))) \
                    -(np.dot(np.sum(np.transpose(cluster_normals)),np.sum(cluster_normals))))

            radius_0.append(r0)

            print(r0,'r0')
            print(radius_0,'radius_0')


            # print(np.sum(cluster_normals,axis = 0),'sumclsuters')
            # print(np.sum(clusters,axis = 1),'sumclsuters')
            # print(np.sum(np.dot(r0,cluster_normals)+(cluster_normals),axis = 0),'dot')

            # print(1/len(cluster_normals),'(1/len(cluster_normals)')


            c0 = np.squeeze((1/len(cluster_normals))*(np.sum(np.dot(r0,cluster_normals)+(clusters),axis = 1)))

            center_0.append(c0)

            print(c0,'c0')
            print(center_0,'center_0')

            x0 = np.array([c0[0], c0[1], c0[2], LA.norm(r0)])

            xstart_0.append(x0)

            print(x0,'x0')
            print(xstart_0,'xstart_0')

            # decision_variabels =
            length_normals = len(cluster_normals)


            f0 = (1/length_normals)*np.sum(np.exp2(LA.norm(np.squeeze(clusters) - c0)- r0))

            jacobian_sphere = []

            # print(np.transpose(clusters),'cluster')
            # print(clusters[0][0][0],'cluster')
            # print(clusters[0][0][1],'cluster')
            # print(np.transpose(clusters)[1,:],'cluster')
            # print(clusters[0][1][1],'cluster')

            cxc = c0[0]
            cyc = c0[1]
            czc = c0[2]
            rc = LA.norm(r0)

            mix = np.squeeze(np.transpose(clusters)[0,:])
            miy = np.squeeze(np.transpose(clusters)[1,:])
            miz = np.squeeze(np.transpose(clusters)[2,:])
            cxc = np.linspace(cxc,cxc,len(clusters[0]))
            cyc = np.linspace(cyc,cyc,len(clusters[0]))
            czc = np.linspace(czc,czc,len(clusters[0]))
            rc = np.linspace(rc,rc,len(clusters[0]))



            # input_vector = np.append(input_vector,mix)
            # input_vector = np.append(input_vector,miy)
            # input_vector = np.append(input_vector,miz)
            # input_vector = np.append(input_vector,cxc)
            # input_vector = np.append(input_vector,cyc)
            # input_vector = np.append(input_vector,czc)
            # input_vector = np.append(input_vector,rc)

            input_vector = []

            input_vector.append(mix)
            input_vector.append(miy)
            input_vector.append(miz)
            input_vector.append(cxc)
            input_vector.append(cyc)
            input_vector.append(czc)
            input_vector.append(rc)

            # print(input_vector,'input_vector')
            # if i == 0:
            #     np.savetxt('camera/jac_input_vector', input_vector)

            jac_input_vector = (input_vector)
            print(i,'i')
            # jac_input_vector = []
            #
            # temp.append(jac_input_vector)

        # print((jac_input_vector),'jac_input_vector')
        #

        # print(jac_input_vector,'jac_input_vector')
        # print(temp,'temp')
        # print((jac_input_vector),'jac_input_vector')
        print(np.shape((jac_input_vector)),'jac_input_vector')

        np.savetxt('camera/jac_input_vector', jac_input_vector, fmt='%f')

        mix = np.squeeze(np.transpose(clusters)[0,:])
        miy = np.squeeze(np.transpose(clusters)[1,:])
        miz = np.squeeze(np.transpose(clusters)[2,:])


        startf0 = 0

        startf0 = np.sqrt((mix - c0[0])**2 + (miy - c0[1])**2  + (miz - c0[2])**2) - r0
        print(np.shape(startf0),'startf0')

        # print(np.shape(temp[0]),'temp')
        # print(np.shape(temp),'temp')

        jacobian_sphere = seg.execute_jacobian(np.transpose(jac_input_vector))

        print('done jacobian_sphere')
        # print(jacobian_sphere,'jacobian_sphere')
        print(np.shape(jacobian_sphere),'jacobian_sphere')
        print(np.shape(jacobian_sphere[0]),'jacobian_sphere len')
        print(length_normals),'length_normals'
        print('done')

        # delta = -(dot(Jt*J)^-1*Jt*f0)

        lm_delta = -np.dot( inv(np.dot(np.transpose(jacobian_sphere),jacobian_sphere)), (np.dot(np.transpose(jacobian_sphere),startf0)))

        mix = np.squeeze(np.transpose(clusters)[0,:])
        miy = np.squeeze(np.transpose(clusters)[1,:])
        miz = np.squeeze(np.transpose(clusters)[2,:])

        eps = 10**(-6)

        lamda = 10
        c = 2

        count = 0
        first_loop = True

        print(LA.norm(lm_delta),'LA.norm(lm_delta)')
        print(eps,'eps')

        print(np.array([c0[0],c0[1] ,c0[2] ,r0]))

        countapp = []
        lm_F_x0_app = []

        while (LA.norm(lm_delta) > eps ):

            count += 1

            if first_loop == True:

                cx  = c0[0]
                cy  = c0[1]
                cz  = c0[2]
                r   = r0

                x_current = np.array([c0[0] ,c0[1] ,c0[2] ,r0 ])

                mix = np.squeeze(np.transpose(clusters)[0,:])
                miy = np.squeeze(np.transpose(clusters)[1,:])
                miz = np.squeeze(np.transpose(clusters)[2,:])
                cxc = np.linspace(cx,cx,len(clusters[0]))
                cyc = np.linspace(cy,cy,len(clusters[0]))
                czc = np.linspace(cz,cz,len(clusters[0]))
                rc = np.linspace(r,r,len(clusters[0]))

                lm_input_vector = []

                lm_input_vector.append(mix)
                lm_input_vector.append(miy)
                lm_input_vector.append(miz)
                lm_input_vector.append(cxc)
                lm_input_vector.append(cyc)
                lm_input_vector.append(czc)
                lm_input_vector.append(rc)

                first_loop = False

            else:

                cx  = x_current[0]
                cy  = x_current[1]
                cz  = x_current[2]
                r   = x_current[3]


                mix = np.squeeze(np.transpose(clusters)[0,:])
                miy = np.squeeze(np.transpose(clusters)[1,:])
                miz = np.squeeze(np.transpose(clusters)[2,:])
                cxc = np.linspace(cx,cx,len(clusters[0]))
                cyc = np.linspace(cy,cy,len(clusters[0]))
                czc = np.linspace(cy,cy,len(clusters[0]))
                rc = np.linspace(r,r,len(clusters[0]))

                lm_input_vector = []

                lm_input_vector.append(mix)
                lm_input_vector.append(miy)
                lm_input_vector.append(miz)
                lm_input_vector.append(cxc)
                lm_input_vector.append(cyc)
                lm_input_vector.append(czc)
                lm_input_vector.append(rc)


            # f0
            lm_f0 = np.sqrt((mix - cx)**2 + (miy - cy)**2  + (miz - cz)**2) - r

            # J  = J(x0)
            lm_jacobian_sphere = seg.execute_jacobian(np.transpose(lm_input_vector))

            # H  = dot(Jt,J)
            lm_hessian = np.dot(np.transpose(lm_jacobian_sphere),lm_jacobian_sphere)

            lm_hessian_lamda_I = np.identity(4) * lamda

            lm_delta = -np.dot(inv(lm_hessian + lm_hessian_lamda_I), (np.dot(np.transpose(lm_jacobian_sphere),lm_f0)))

            #F error first round
            lm_F_x0 = np.dot(np.transpose(lm_f0),lm_f0) + 2 * np.dot(np.dot(np.transpose(lm_f0),lm_jacobian_sphere) , lm_delta) + np.dot(np.transpose(lm_delta),np.dot(lm_hessian,lm_delta))


            delta_x0 = np.array([cx + lm_delta[0], cy + lm_delta[1], cz + lm_delta[2], r + lm_delta[3]])

            lm_f0_delta = np.sqrt((mix - delta_x0[0])**2 + (miy - delta_x0[1])**2  + (miz - delta_x0[2])**2) - delta_x0[3]


            #F error
            lm_F_x0_delta = np.dot(np.transpose(lm_f0_delta),lm_f0_delta) + 2 * np.dot(np.dot(np.transpose(lm_f0_delta),lm_jacobian_sphere)  , lm_delta) + np.dot(np.transpose(lm_delta),np.dot(lm_hessian,lm_delta))



            print('#############################################################')
            print(count,'count')

            print(lm_F_x0,'lm_F_x0')
            # print(lm_f0_delta,'lm_f0_delta')
            print(lm_F_x0_delta,'lm_F_x0_delta')
            # print(delta_x0,'delta_x0')
            # print(lm_delta,'lm_delta')
            # # f0
            # lm_f0 = np.sqrt((mix - delta_x0[0])**2 + (miy - delta_x0[1])**2  + (miz - delta_x0[2])**2) - delta_x0[3]
            #
            # lm_F_x0_delta = np.dot(np.transpose(lm_f0),lm_f0) + 2 * np.dot(np.dot(np.transpose(lm_f0),lm_jacobian_sphere) , lm_f0_delta) + np.dot(np.transpose(lm_delta),np.dot(lm_hessian,lm_delta))


            print(x_current,'x_current')

            if ( lm_F_x0_delta < lm_F_x0):
                x_current = x_current + lm_delta
                lamda = lamda/c
            else:
                lamda = c * lamda

            print(lamda,'lamda')
            print(c,'c')
            print(x_current,'x_current')
            print(LA.norm(lm_delta),'LA.norm(lm_delta)')

            countapp.append(count)
            lm_F_x0_app.append(lm_F_x0)

            if count == 40:
                plt.axis([0, 40, 0, 1])
                plt.scatter(countapp, lm_F_x0_app)
                plt.pause(0.05)



        print('done??')
        exit()

        ### loop to look at each cluter 1 by 1
        for i in range(np.amax(clusterlabels)+1):
            ###########################################################
            #Cylinders

            ## calculate normal centroid
            nbar = np.sum(cluster_normals,axis = 0)
            nbar = nbar/np.linalg.norm(nbar)

            print(nbar,'nbar')

            ## initialize z axis
            z_hat = np.array([0,0,-1])

            #compute the alignment transofrmation with the z axis
            r_theta = np.arccos(np.dot(nbar,z_hat))
            r_A = np.cross(nbar,z_hat)

            ## Rodrigues formulation for rmat
            skew_w = np.array([[0,-r_A[2],r_A[1]],[r_A[2],0,-r_A[0]],[-r_A[1],r_A[0],0]])

            r_mat = np.eye(3) + skew_w + (skew_w.dot(skew_w))/(1+np.cos(r_theta))

            ## normals of cluster aligned with z axis
            nj_alligned = np.dot(np.squeeze(r_mat),np.transpose(cluster_normals))

            ## compute aligend normal centroid
            nja = np.mean(nj_alligned,axis = 1)
            nja = nja/np.linalg.norm(nja)

            print(nja,'nja')

            ## decompose the current cluster for better computation
            decomp_cluster = np.transpose(nj_alligned)

            m = np.sqrt(len(decomp_cluster))

            delta = np.exp2(np.sum(np.exp2(decomp_cluster[0,:]) - \
                                   np.exp2(decomp_cluster[1,:]))) + \
                                   4 * (np.exp2(np.sum( decomp_cluster[0,:] * \
                                                       decomp_cluster[1,:])))

            ## compute max and min angle for quardratic solution
            theta_min = np.arctan2((np.sum(np.exp2(decomp_cluster[1,:]) -\
                                           np.exp2(decomp_cluster[0,:])) - \
                                           np.sqrt(delta)) ,\
                                   (2 * np.sum(decomp_cluster[0,:] * (decomp_cluster[1,:]))))
            print(theta_min,'theta_min')

            a_min = np.array([[np.cos(theta_min)],[np.sin(theta_min)],[0]])

            print(a_min,'a_min')

            ## Rodrigues formulation for rmat
            skew_w = np.array([[0,-r_A[2],r_A[1]],[r_A[2],0,-r_A[0]],[-r_A[1],r_A[0],0]])

            ## for inverse rodrigues take the inverse of the matrix ez
            r_mat = np.eye(3) + skew_w + (skew_w.dot(skew_w))/(1+np.cos(r_theta))




            ## inital radius guess
            r0 = -(np.dot(len(cluster_normals), np.transpose(np.sum(clusters)) * np.sum(cluster_normals)) \
                   - np.dot((np.transpose(np.sum(clusters))),np.sum(cluster_normals))) \
                    /np.dot(len(cluster_normals),(np.sum(np.dot(np.transpose(cluster_normals),cluster_normals))) \
                    -(np.dot(np.sum(np.transpose(cluster_normals)),np.sum(cluster_normals))))


        # center = [0.09390587, 0.06827942, 0.61555819 ]
        # # [0.09956031 0.07175935 0.99180109]
        # radius_sphere = 0.035
        #
        # mesh = o3d.geometry.TriangleMesh.create_sphere( radius_sphere )

        return mesh, center


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
    draw_aruco_corner = False
    draw2d_aruco_marker = False
    read_from_file = True


    while 1:

        #compute the rgb and depth img
        rgb_img, depth_img = cam.stream(colored_depth=False)

        #generate pcd
        xyz = cam.generate_pcd(depth_img)

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
        seg.non_parametric_surface_class(binxyz, labels, neighbors)

        # seg.segmentate(crxyz,normals,visualize_db_clusters)
        # input_vector = np.loadtxt('camera/jac_input_vector')
        # jacobian_sphere = seg.execute_jacobian((input_vector))
        # print('done')

        ########### not in use yet ###########
        mesh, center = seg.surface_estimation(binxyz, labels, neighbors)

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

        # check sphere center
        # print(f'Center of mesh: {mesh.get_center()}')
        # print(center,'center')
        #
        # pcdcenter = pcd.get_center()
        #
        # print(f'Center of pcd: {pcd.get_center()}')
        #
        # center2 = mesh.get_center() - pcd.get_center()
        #
        # mesh1 = mesh.translate((-center2[0],-center2[1],-center2[2]))
        # # mesh1 = mesh.translate((center[0],center[1],center[2]))
        #
        # print(f'Center of mesh1: {mesh1.get_center()}')
        #
        # print(f'Center of mesh: {mesh.get_center()}')

        ## visualize point cloud caculated from the depth image
        if added == True:
            vis.add_geometry(pcd)
            added = False
        vis.update_geometry(pcd)
        # o3d.visualization.draw_geometries([pcd,mesh1])
        vis.poll_events()
        vis.update_renderer()
