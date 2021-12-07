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

class Segmentation:

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

        #keep the smalles normals
        normals = v[:,1]

        #flip normals so they are all in the same direction
        if normals.dot(xm)>0:
            normals = -normals

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
        normals = v[:,0]

        #flip normals so they are all in the same direction
        if normals.dot(xm)>0:
            normals = -normals

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

            temp_index = np.where(Cosine_Similarity[i] > 0.75)

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
            legend1 = ax.legend(*scatter.legend_elements(num=len(labels_unq)-1), loc="upper left", title="Ranking")
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

                    np.savetxt('camera/frame.txt', cam2markerHom)
        if read_from_file == True:
            cam2markerHom = np.loadtxt('camera/frame.txt')


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
        # Best
        # bin_width    =  0.34 #M real is 0.34 #0.22
        # bin_length   =  0.44 #M real is 0.44 #0.39
        # bin_height   =  0.5 #M real is 0.2 #0.175
        #
        # condt = np.where((markerpcd[0] >= -0.05)  & (markerpcd[0] <= bin_width) &
        #                  (markerpcd[1] <= -0.05) & (markerpcd[1] >= -bin_length) &
        #                  (markerpcd[2] >= 0.19) & (markerpcd[2] <= bin_height) )

        ##############################################
        # Best
        bin_width    =  0.28 #M real is 0.34 #0.22
        bin_length   =  0.44 #M real is 0.44 #0.39
        bin_height   =  0.5 #M real is 0.2 #0.175

        condt = np.where((markerpcd[0] >= -0.075)  & (markerpcd[0] <= bin_width) &
                         (markerpcd[1] <= -0.075) & (markerpcd[1] >= -bin_length) &
                         (markerpcd[2] >= 0.185) & (markerpcd[2] <= bin_height) ) ## 0.19 is really good

        xyz = xyz[condt]


        if draw2d == True:
            ########show 2d circle on the corner of the marker useing the 3d pose
            bin_corner = np.dot(cam2markerHom,marker_bin_corner)
            bin_corner_2d = (np.dot(cam.camera_mat , bin_corner[:3])/bin_corner[2][0])[:2]
            cv2.circle(rgb_img, (int(bin_corner_2d[0]),int(bin_corner_2d[1])), radius=5, color=(0, 0, 255), thickness=2)
            cv2.imshow("Image1",rgb_img)
            cv2.waitKey(1)

        return xyz

    def surface_estimation(self, xyz, clusterlabels, neighbors):

        clusters = []
        cluster_idx = []
        cov_mat = []
        d = []

        for i in range(np.amax(clusterlabels)+1):

            cluster_idx.append(np.where((clusterlabels == i)))

            clusters.append(xyz[cluster_idx[i]])


            cluster_mean = np.mean(clusters[i])

            ###########################################################
            #plane
            cov_mat.append( (1/len(clusters[i])) *
                           (np.dot(np.transpose(clusters[i] - cluster_mean),
                                                (clusters[i] - cluster_mean))))

            #print(cov_mat,'cov_mat')

            normals = np.asarray(p.map(seg.normals,cov_mat))

            #d = distance to origin calculated from the mean position
            d.append(np.dot( -normals, cluster_mean))


            #plane error missing


            ###########################################################
            #sphere

            #print(np.asarray([clusters[i]]),'clusters')

            normals = np.asarray(p.map(seg.normals,np.asarray([clusters[i]])))
            #normals = p.map(seg.normals,clusters[i])

            print(normals,'normals')


            r0 = -(np.dot(normals[i], np.dot(np.transpose( np.sum(clusters[i]),np.sum(normals[i])))))
            print(r0,'r0')

            r0 = -(np.dot(normals[i],np.sum(np.dot(np.transpose(clusters[i]),normals[i])))) \
                   - np.dot(np.sum(np.transpose(clusters[i])),np.sum(normals[i])) \
                    /(np.dot(np.sum(np.dot(np.transpose(normals[i]),normals[i])))) \
                    - (np.sum(np.dot(np.sum(np.transpose(normals[i])),np.sum(normals[i]))))

            print(r0,'r0')

        return

    def non_parametric_surface_class(self, xyz, clusterlabels, neighbors):

        clusters = []
        cluster_idx = []
        cluster_mean = []
        ni = []
        mi = []
        principal_norals = []
        Cosine_Similarity_primitive = []
        shape_guess = []

        labels_unq = np.unique(clusterlabels)
        print(labels_unq,'labels_unq')
        # print(clusterlabels,'clusterlabels')
        # print(range(np.amax(clusterlabels)),'range')

        for i in range(np.amax(clusterlabels)+1):

            print(i,'i')

            cluster_idx.append(np.where((clusterlabels == i)))

            clusters = []

            clusters.append(xyz[cluster_idx[i]])

            # print(clusters,'clusters')

            tree = KDTree(np.squeeze(clusters))
            dist, ind = tree.query(np.squeeze(clusters), k = neighbors)

            clusters_ind = np.squeeze(clusters)[ind]

            # print(clusters_ind,'clusters_ind')
            # print(len(clusters_ind),'clusters_ind')
            # print(len(clusters_ind[0]),'clusters_ind')

            #print(clusters_ind,'clusters_ind')

            # seg.surface_normal_estimation(xyz[cluster_idx[i]], 30, True)

            cluster_normals = np.asarray(p.map(seg.normals,clusters_ind))

            # print(len(cluster_normals),'cluster_normals')

            ######### guassian image ###########
            # seg.gaussian_image(cluster_normals)

            #print(len(cluster_normals[:,0]),'cluster_normals')
            # print(len(cluster_normals),'cluster_normals')
            # print(len(cluster_normals[0]),'cluster_normals')

            #compute the z axis and center of patches
            ni = []
            ni.append(np.mean(cluster_normals[:,0]))
            ni.append(np.mean(cluster_normals[:,1]))
            ni.append(np.mean(cluster_normals[:,2]))

            print(ni,'ni')

            z_hat = np.array([0,0,1])


            #print(ni,'ni')
            #print(len(ni),'ni')

            #compute the alignment transofrmation with the z axis
            r_theta = np.degrees(np.arccos(np.dot(np.transpose(ni),z_hat)))
            r_A = np.cross(ni,z_hat)

            #print(r_A,'r')
            #print(r_theta,'r_theta')

            en      = np.cos(r_theta) + np.exp2(r_A[0])*(1-np.cos(r_theta))
            to      = r_A[0]*r_A[1]*(1-np.cos(r_theta)) - r_A[2]*np.sin(r_theta)
            tre     = r_A[0]*r_A[2]*(1-np.cos(r_theta)) + r_A[1]*np.sin(r_theta)
            fire    = r_A[1]*r_A[0]*(1-np.cos(r_theta)) + r_A[2]*np.sin(r_theta)
            fem     = np.cos(r_theta) + np.exp2(r_A[0])*(1-np.cos(r_theta))
            seks    = r_A[1]*r_A[2]*(1-np.cos(r_theta)) - r_A[0]*np.sin(r_theta)
            syv     = r_A[2]*r_A[0]*(1-np.cos(r_theta)) - r_A[1]*np.sin(r_theta)
            otte    = r_A[2]*r_A[1]*(1-np.cos(r_theta)) + r_A[0]*np.sin(r_theta)
            ni      = np.cos(r_theta) + np.exp2(r_A[2])*(1-np.cos(r_theta))

            r_mat = np.array([[en, to, tre],
                              [fire, fem, seks],
                              [syv, otte, ni]])

            #print(r_mat,'r_mat1')

            nj_alligned = np.dot(np.squeeze(r_mat),np.transpose(cluster_normals))

            # print(np.shape(nj_alligned[0,:]),'nj_alligned1')
            #print((nj_alligned),'nj_alligned2')
            # print(len(nj_alligned[0]),'nj_alligned3')

            nja = []
            nja.append(np.mean(nj_alligned[0,:]))
            nja.append(np.mean(nj_alligned[1,:]))
            nja.append(np.mean(nj_alligned[2,:]))

            #print(nja,'nja')
            #plt.scatter(nj_alligned[0,:],nj_alligned[1,:])
            #plt.show()

            #print(np.delete(nj_alligned,2,0),'size')
            # print(np.shape(np.vstack(nj_alligned[0,:])),'xj')
            # print(np.shape(np.vstack(nj_alligned[1,:])),'yj')
            # print(np.dot(nj_alligned[0,:] , nj_alligned[1,:]),'dot')
            # print(np.sum(nj_alligned[0,:] * nj_alligned[1,:]),'*')

            decomp_cluster = np.squeeze(np.delete(nj_alligned,2,0))

            decomp_cluster = np.transpose(np.asarray([decomp_cluster[0,:],decomp_cluster[1,:]]))

            # print(decomp_cluster,'decomp_cluster')

            neighbors_decomp = 225

            if (len(decomp_cluster)< 225):
                neighbors_decomp = len(decomp_cluster)

            #print(neighbors_decomp,'neighbors_decomp')

            tree = KDTree(decomp_cluster)
            dist, ind = tree.query(decomp_cluster, k = neighbors_decomp)

            decomp_cluster = np.asarray(decomp_cluster[ind])

            # print(len(decomp_cluster[0][0]),'len')
            # #print(decomp_cluster[0][:],'decomp_cluster')
            # print(decomp_cluster,'decomp_cluster')


            cluster_concl = []
            Np = []
            ## 0.000056 5.6e-05
            eth = 0.000071
            print(eth,'eth')
            unique = []
            counts = []

            mse_max_app = []
            mse_min_app = []

            # print(clusters,'clusters')
            #
            # print(clusters[0][:,0],'clusters')

            #clusters0 = clusters[:,0]

            mi = []
            mi.append(np.mean((clusters[0][:,0])))
            mi.append(np.mean((clusters[0][:,1])))
            mi.append(np.mean((clusters[0][:,2])))

            ni = []
            ni.append(np.mean(cluster_normals[:,0]))
            ni.append(np.mean(cluster_normals[:,1]))
            ni.append(np.mean(cluster_normals[:,2]))


            for j in range(len(decomp_cluster)):
                ############### Principal Curvatures ####################

                decomp_cluster_temp = decomp_cluster[j][:]

                # print(j,'j')
                # print(decomp_cluster_temp[0][0],'decomp_cluster_temp')

                m = np.sqrt(len(decomp_cluster_temp))
                #print(m,'m')

                delta = np.exp2(np.sum(np.exp2(decomp_cluster_temp[0,:]) - \
                                       np.exp2(decomp_cluster_temp[1,:]))) + \
                                       4 * (np.exp2(np.sum( decomp_cluster_temp[0,:] * \
                                                           decomp_cluster_temp[1,:])))

                theta_max = np.arctan((np.sum(np.exp2(decomp_cluster_temp[1,:]) - \
                                               np.exp2(decomp_cluster_temp[0,:])) + \
                                               np.sqrt(delta)) /\
                                       (2 * np.sum(decomp_cluster_temp[0,:] * (decomp_cluster_temp[1,:]))))

                theta_min = np.arctan2((np.sum(np.exp2(decomp_cluster_temp[1,:]) -\
                                               np.exp2(decomp_cluster_temp[0,:])) - \
                                               np.sqrt(delta)) ,\
                                       (2 * np.sum(decomp_cluster_temp[0,:] * (decomp_cluster_temp[1,:]))))

                mse_max = (1/np.exp2(m)) * \
                          np.sum(np.exp2( (decomp_cluster_temp[0,:] * np.cos(theta_max)) + \
                                          (decomp_cluster_temp[1,:] * np.sin(theta_max)) ))

                mse_min = (1/np.exp2(m)) * \
                          np.sum(np.exp2( (decomp_cluster_temp[0,:] * np.cos(theta_min)) + \
                                          (decomp_cluster_temp[1,:] * np.sin(theta_min)) ))

                # print(delta,'delta')
                # print(np.sqrt(delta),'delta')
                # print(theta_max,'theta_max')
                # print(theta_min,'theta_min')
                # print(mse_max,'mse_max')
                # print(mse_min,'mse_min')

                mse_max_app.append(mse_max)
                mse_min_app.append(mse_min)


                ####real
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

                # ni # center normal
                # cluster_normals # nj all cluster_normals
                # clusters # mj all

                dj = np.array(clusters) - np.array(mi)

                # print(cluster_normals,'cluster_normals')
                # print(ni,'ni')

                test_convexity = np.transpose(np.cross(np.cross(cluster_normals,ni),dj))

                if (test_convexity >= 0):
                    convexity = 1 ## convex
                elif (test_convexity <= 0):
                    convexity = 0 ## concave



            # print(mi,'mi')
            # print(dj,'dj')
            # print(convexity,'convexity')


            # print(np.mean(mse_max_app),'mse_max_app')
            # print(np.mean(mse_min_app),'mse_min_app')
            #
            # print(cluster_concl,'cluster_concl')

            unique, counts = np.unique(cluster_concl, return_counts=True)

            # print(unique,'unique')
            # print(counts,'counts')

            shape_guess.append(np.array(unique[np.where(counts == np.amax(counts))]))

            # print(np.squeeze(shape_guess),'shape_guess')



        return



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

        ########### not in use yet ###########
        #seg.surface_estimation(binxyz, labels, neighbors)

        #visualize the pcd
        pcd.points = o3d.utility.Vector3dVector(crxyz)

        #Transfor viewpoint to look form the camera perspective
        pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

        #visualize the depth_img with color/gray scale
        #rgb_img, depth_img = cam.stream(colored_depth=True)

        ## visualize rgb and depth image
        #cv2.imshow("rgb", rgb_img)
        #cv2.imshow("depth", depth_img)
        #cv2.waitKey(1)

        ## visualize point cloud caculated from the depth image
        if added == True:
            vis.add_geometry(pcd)
            added = False
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
