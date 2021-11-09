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

    def surface_normal_estimation(self, xyz, neighbors,   test):

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
        db = DBSCAN(eps=0.022, min_samples=50).fit(xyz)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        #save the clusters and the noise from the scan
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)


        #visualize the clusters
        if visualize == True:
            fig = plt.figure()
            ax = plt.axes(projection='3d')

            ax.scatter3D(xyz[:,0],xyz[:,1],xyz[:,2], c=db.labels_, cmap='jet')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            plt.show()

        return labels

    def bin_removal(self,xyz, rgb_img, draw, draw2d):
        #this function removes the points of the bin
        #this is done by est the pose of a aruco marker and useing the size of the bin
        #the pound cloud data points are transformed to be seen from the bin and
        #then compared to the length width and height of the bin

        #define marker finder parameters and find marker
        marker_size = 0.105 #size of marker in M
        type_arucodict = cv2.aruco.DICT_ARUCO_ORIGINAL
        arucoDict = cv2.aruco.Dictionary_get(type_arucodict)
        arucoParams = cv2.aruco.DetectorParameters_create()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(rgb_img, arucoDict,
        	parameters=arucoParams)


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

        bin_width    =  0.22 #M real is 0.34
        bin_length   =  0.39 #M real is 0.44
        bin_height   =  0.175 #M real is 0.2

        condt = np.where((markerpcd[0] >= -0.045)  & (markerpcd[0] <= bin_width) &
                         (markerpcd[1] <= 0.05) & (markerpcd[1] >= -bin_length) &
                         (markerpcd[2] >= 0.05) & (markerpcd[2] <= bin_height) )

        xyz = xyz[condt]


        if draw2d == True:
            ########show 2d circle on the corner of the marker useing the 3d pose
            bin_corner = np.dot(cam2markerHom,marker_bin_corner)
            bin_corner_2d = (np.dot(cam.camera_mat , bin_corner[:3])/bin_corner[2][0])[:2]
            cv2.circle(rgb_img, (int(bin_corner_2d[0]),int(bin_corner_2d[1])), radius=5, color=(0, 0, 255), thickness=2)
            cv2.imshow("Image1",rgb_img)
            cv2.waitKey(1)

        return xyz

    def surface_estimation(self, xyz, clusterlabels):

        clusters = np.array([np.amax(clusterlabels)])
        cluster_idx = 0

        print(type(clusterlabels),'clusterlabels')
        print(clusterlabels,'clusterlabels')

        i = np.arange(np.amax(clusterlabels))

        cluster_idx = np.where((clusterlabels == i))

        clusters[i] = xyz[cluster_idx]

        #for i in range(np.amax(clusterlabels)):

            #cluster_idx = np.where(clusterlabels = i)

        print(clusters,'clusters')

        exit()

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
    visualize_db_clusters = False
    draw_aruco_corner = False
    draw2d_aruco_marker = False

    while 1:
        #compute the rgb and depth img
        rgb_img, depth_img = cam.stream(colored_depth=False)



        #generate pcd
        xyz = cam.generate_pcd(depth_img)

        #neighbor filtering of pcd
        FNxyz = seg.filter_neighbors(xyz, neighbors)

        #radius filtering of pcd
        FRxyz = seg.filter_radius(FNxyz, radius, minpoints,visualize_Filter_Radius)

        #surface normals estimation of pcd and alignment
        normals = seg.surface_normal_estimation(FRxyz, neighbors, test_surface_normals)

        #cam.Gaussian_image(normals)

        #Crease_removal of the pcd
        CRxyz = seg.crease_removal(FRxyz,normals,neighbors)

        #surface normals estimation of pcd and alignment to see if the result is better
        normals = seg.surface_normal_estimation(CRxyz, neighbors, test_surface_normals)

        #remove bin from point cloud
        binxyz = seg.bin_removal(CRxyz,rgb_img, draw = draw_aruco_corner, draw2d = draw2d_aruco_marker)

        #clustering/segmentate of pcd
        labels = seg.segmentate(binxyz,normals,visualize_db_clusters)

        seg.surface_estimation(binxyz, labels)

        #visualize the pcd
        pcd.points = o3d.utility.Vector3dVector(binxyz)

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
