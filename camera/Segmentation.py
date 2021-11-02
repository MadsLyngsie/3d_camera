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

    def FilterNeighbors(self, xyz, neighbors):
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

    def FilterRadius(self,xyz,radius,minpoints,visualize):

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

    def SurfaceNormalEstimation(self, xyz, neighbors,   test):

        #tree query for neighbors
        tree = KDTree(xyz)
        dist, ind = tree.query(xyz, k = neighbors)

        #make centorid indices
        Xi = xyz[ind]

        #multiprocessing the calculation of the normals
        normals = p.map(seg.Normals,Xi)

        #test to show normals calculated maunally
        if test == True:
            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points  = o3d.utility.Vector3dVector(xyz)
            self.pcd.normals = o3d.utility.Vector3dVector(normals)
            o3d.visualization.draw_geometries( [self.pcd], point_show_normal=True)

        return normals

    @staticmethod
    def Normals(Xi):


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

    def Gaussian_image(self, normals):

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

    def Crease_removal(self,xyz,normals,neighbors):

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

    def Segmentate(self, xyz, normals, visualize):

        #doing a dbscan to attemp a clustering of points in the pcd
        db = DBSCAN(eps=0.028, min_samples=50).fit(xyz)
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

        return

    def marker_detection(self, rgb_img):

        type_arucodict = cv2.aruco.DICT_ARUCO_ORIGINAL
        arucoDict = cv2.aruco.Dictionary_get(type_arucodict)
        arucoParams = cv2.aruco.DetectorParameters_create()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(rgb_img, arucoDict,
        	parameters=arucoParams)


        draw = False
        if draw == True:
            if len(corners) > 0:
                ids = ids.flatten()

                for (markerCorner, markerID) in zip(corners, ids):
                    # extract the marker corners (which are always returned in

                    # top-left, top-right, bottom-right, and bottom-left order)
                    corners = markerCorner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corners

                    # convert each of the (x, y)-coordinate pairs to integers
                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))

                    #print(topLeft,'topLeft')
                    #print(topRight,'topRight')
                    #print(bottomRight,'bottomRight')
                    #print(bottomLeft,'bottomLeft')

                    #drawline around marker
                    cv2.line(rgb_img,topLeft,topRight,(255,0,255),2)
                    cv2.line(rgb_img,topRight,bottomRight,(0,0,255),2)
                    cv2.line(rgb_img,bottomRight,bottomLeft,(255,0,0),2)
                    cv2.line(rgb_img,bottomLeft,topLeft,(0,255,0),2)
                    cv2.imshow("Image",rgb_img)
                    cv2.waitKey(0)


        #calculate pose
        ArUCoSize = (2,2)
        #ArUCoSize = (7,7)#chess


        objp = np.array([[0,0,0],[0,0.1,0],[0.1,0.1,0],[0.1,0,0]])
        #objp = np.zeros((ArUCoSize[0] * ArUCoSize[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:ArUCoSize[0], 0:ArUCoSize[1]].T.reshape(-1,2)
        #axis = np.float32([3,0,0], [0,3,0, [0,0,-3]]).reshape(-1,3)
        axisbox = np.float32([[0,0,0], [0,1,0], [1,1,0], [1,0,0],
                                [0,0,-1], [0,1,-1],[1,1,-1], [1,0,-1]])


        #img = cv2.imread(rgb_img)
        #gray = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2GRAY)

        #print(np.array([corners]),'cornersAruco')
        #ret, corners = cv2.findChessboardCorners(gray, (7,7),None)

        #print(ret,'ret')
        #print(corners,'cornerschess')

        #corners = np.array([[topLeft[0],topLeft[1],[topRight[0],topRight[1]],[bottomRight[0], bottomRight[1]],
        #            [bottomLeft[0],bottomLeft[1]]]])

        corners = np.array(corners)
        print(corners,'corners')

        #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        #corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        #print(corners2,'corners2')

        ret, rvecs ,tvecs = cv2.solvePnP(objp, corners, self.camera_mat, self.distCoeffs)
        imgpts, jac = cv2.projectPoints(axisbox, rvecs, tvecs, self.camera_mat, self.distCoeffs)

        #print(imgpts,'imgpts')
        #print(jac,'jac')

        print(imgpts,'imgpts')

        imgpts = np.int32(imgpts).reshape(-1,2)

        print(imgpts,'imgpts2')

        Draw = True
        if Draw == True:
            img = cv2.drawContours(rgb_img, [imgpts[:4]],-1,(0,255,0),-3)

            for i,j in zip(range(4),range(4,8)):
                img = cv2.line(rgb_img, tuple(imgpts[i]),tuple(imgpts[j]),(255),3)

            img = cv2.drawContours(rgb_img, [imgpts[:4]],-1,(0,0,255),3)
            cv2.imshow('img',img)
            cv2.waitKey(0)




        exit()
        input()

        return


if __name__ == '__main__':
    cam = Camera()
    seg = Segmentation()
    p = Pool()

    pcd = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()

    #vis.create_window("Point Clouds", width=640, height=480)
    added = True

    neighbors = 30
    radius = 0.02
    minpoints = 30
    test_surface_normals = False
    Visualize_Filter_Radius = False
    Visualize_db_clusters = True

    while 1:
        #compute the rgb and depth img
        rgb_img, depth_img = cam.stream(colored_depth=False)

        #seg.marker_detection(rgb_img)

        #generate pcd
        xyz = cam.generate_pcd(depth_img)

        #neighbor filtering of pcd
        FNxyz = seg.FilterNeighbors(xyz, neighbors)

        #radius filtering of pcd
        FRxyz = seg.FilterRadius(FNxyz, radius, minpoints,Visualize_Filter_Radius)

        #surface normals estimation of pcd and alignment
        normals = seg.SurfaceNormalEstimation(FRxyz, neighbors, test_surface_normals)

        #cam.Gaussian_image(normals)

        #Crease_removal of the pcd
        CRxyz = seg.Crease_removal(FRxyz,normals,neighbors)

        #surface normals estimation of pcd and alignment to see if the result is better
        normals = seg.SurfaceNormalEstimation(CRxyz, neighbors, test_surface_normals)

        #clustering/segmentate of pcd
        seg.Segmentate(CRxyz,normals,Visualize_db_clusters)

        #visualize the pcd
        pcd.points = o3d.utility.Vector3dVector(CRxyz)

        #visualize the depth_img with color/gray scale
        rgb_img, depth_img = cam.stream(colored_depth=True)

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
