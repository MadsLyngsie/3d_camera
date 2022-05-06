import sympy as sp
import numpy as np
import open3d as o3d
from sympy.utilities.autowrap import autowrap
import matplotlib.pyplot as plt
from SO3 import SO3
from SE3 import SE3

print("Environment Ready")
# ------------------------------------>
def jacobian_cylinder():
    # Notation:
    # M_i = (m_ix, m_iy, m_iz) : range point M_i
    # d                        : distance from camera origin O to closest point P on cylinder's axis
    # theta                    : polar coordinate of OP
    # phi                      : polar coordinate of OP
    # alpha                    : angle between cylinder's axis A and N_theta
    # r                        : cylinder radius r
    # range point M_i
    m_ix, m_iy, m_iz         = sp.symbols('m_ix, m_iy, m_iz')
    M_i                      = sp.Matrix([m_ix, m_iy, m_iz])
    # cylinder parameters
    d, theta, phi, alpha, r  = sp.symbols('d, theta, phi, alpha, r')
    # closest point P on cylinder axis to camera origin
    N          = sp.Matrix([sp.cos(phi)*sp.sin(theta), sp.sin(phi)*sp.sin(theta), sp.cos(theta)]) # column vector
    P          = d*N                                                            # column vector
    # cylinder axis
    N_theta    = sp.Matrix([sp.cos(phi)*sp.cos(theta), sp.sin(phi)*sp.cos(theta), -sp.sin(theta)])
    N_phi      = sp.Matrix([-sp.sin(phi), sp.cos(phi), 0])
    A          = N_theta*sp.cos(alpha) + N_phi*sp.sin(alpha)
    # distance from range element M_i to estimated cylinder surface
    D_i        = P + A*(M_i - P).dot(A) - M_i
    e_cyl      = sp.sqrt(D_i[0]**2 + D_i[1]**2 + D_i[2]**2) - sp.sign(r)*r
    # jacobian of loss function
    J_d        = sp.diff(e_cyl, d)
    J_theta    = sp.diff(e_cyl, theta)
    J_phi      = sp.diff(e_cyl, phi)
    J_alpha    = sp.diff(e_cyl, alpha)
    J_r        = sp.diff(e_cyl, r)
    J_cylinder = sp.Matrix([J_d, J_theta, J_phi, J_alpha, J_r])
    J_cylinder = J_cylinder.subs(sp.diff(sp.sign(r), r), 0)
    J_cylinder = autowrap(J_cylinder, backend="f2py", args = [m_ix, m_iy, m_iz, d, theta, phi, alpha, r])
    return J_cylinder

def compute_jacobian_cylinder(lambdified_jacobian_cylinder, input_vector):
    result  = lambdified_jacobian_cylinder(*input_vector).reshape((5,)).astype('float32')
    return result

def compute_distances_to_cylinder(model, cluster_xyz, clipping_distance = 0.003):
    no_pts               = cluster_xyz.shape[0]
    d                    = model[0, 0]
    theta                = model[1, 0]
    phi                  = model[2, 0]
    alpha                = model[3, 0]
    r                    = model[4, 0]
    N                    = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
    N_theta              = np.array([np.cos(phi)*np.cos(theta), np.sin(phi)*np.cos(theta), -np.sin(theta)])
    N_phi                = np.array([-np.sin(phi), np.cos(phi), 0])
    P                    = d*N
    A                    = N_theta * np.cos(alpha) + N_phi * np.sin(alpha)
    print(A,'A')
    # signed distances from xyz_s to model's surface
    print(np.shape(cluster_xyz),'cluster_xyz')
    print(np.shape(P),'P')
    print(P,'P')
    print(A,'A')
    print(np.shape(A),'A')

    temp                 = np.sum((cluster_xyz - P)*A, axis = 1).reshape((no_pts, 1))

    D                    = P + temp.dot(A.reshape((1, 3))) - cluster_xyz
    signed_distances     = (np.linalg.norm(D, axis = 1) - abs(r)).reshape((no_pts, 1))
    # print(signed_distances,'signed_distances')

    lost                 = np.sum(signed_distances*signed_distances) / no_pts
    rmse                 = np.sqrt(lost)
    distances            = np.absolute(signed_distances)
    inliers              = np.argwhere(distances <= clipping_distance)[:, 0]
    adherence            = inliers.shape[0] / no_pts
    return signed_distances, lost, rmse, adherence, inliers

    # cyl_error, cyl_quadratic_lost, rmse, _, _ \
    #                = compute_distances_to_cylinder(cyl_init_model, cluster_xyz)


def initiating_cylinder(cluster_info):
    cluster_xyz                 = cluster_info[0]
    cluster_normals             = cluster_info[1]
    no_pts                      = cluster_xyz.shape[0]
    # -----------------------------> fitting cylinder
    # central normals of cluster
    N_i                            = np.mean(cluster_normals, axis = 0)
    #print(    N_i  ,'    N_i  ')
    N_i                            = N_i / np.linalg.norm(N_i)
    #print(    N_i  ,'    N_i  ')
    Z                              = np.array([0, 0, -1])
    # transformation that aligns N_i with Z-axis
    theta                          = np.arccos(np.dot(N_i, Z))

    A                              = np.cross(N_i, Z) # rotational axis
    theta_r                        = np.array([theta, A[0], A[1], A[2]])
    #print(theta_r,'theta_r')
    R                              = SO3(axis_angle = theta_r).R
    R_inv                          = SO3(axis_angle = theta_r).inverse().R
    reoriented_cluster_normals     = np.dot(R, cluster_normals.transpose())
    reoriented_cluster_normals     = reoriented_cluster_normals.transpose()
    #print(reoriented_cluster_normals,'reoriented_cluster_normals')
    projection_XY                  = reoriented_cluster_normals[:, :2]
    #print(projection_XY,'projection_XY')
    # ---------> initial estimation of cylinder axis using PCA analysis on projection_XY
    k                              = projection_XY.shape[0]
    cov                            = (1/k)*(projection_XY.T).dot(projection_XY)
    #print(cov,'cov')
    w, v                           = np.linalg.eig(cov) # eigenvalues eigenvector analysis
    idx                            = w.argsort()[::-1]
    w                              = w[idx]
    v                              = v[:,idx]
    #print(v,'v')
    A0_projection_XY               = np.array([v[0,1], v[1,1], 0])
    #print(A0_projection_XY,'A0_projection_XY')
    A0                             = np.dot(R_inv, A0_projection_XY)


    #print(A0,'A0')
    #print(Z,'z')
    if A0[2] > 0:
        Z = np.array([0, 0, 1])
    else:
        Z = np.array([0, 0, -1])
    #print(A0,'A0')
    #print(Z,'z')

    # --------------------> initial estimation of cylinder radius and center
    # transformation that aligns A0 with the Z-axis


    theta                           = np.arccos(np.dot(A0, Z))
    A                               = np.cross(A0, Z) #rotational axis
    theta_r                         = np.array([theta, A[0], A[1], A[2]])
    R                               = SO3(axis_angle = theta_r).R
    R_inv                           = SO3(axis_angle = theta_r).inverse().R
    reoriented_cluster_normals      = np.dot(R, cluster_normals.transpose())
    reoriented_cluster_normals      = reoriented_cluster_normals.transpose()
    reoriented_cluster_normals[:,2] = 0

    reoriented_cluster_xyz          = np.dot(R, cluster_xyz.transpose())
    reoriented_cluster_xyz          = reoriented_cluster_xyz.transpose()
    reoriented_cluster_xyz[:,2]     = 0

    #print(np.shape(reoriented_cluster_xyz),'np.shape(reoriented_cluster_xyz)')
    #print(np.shape(reoriented_cluster_normals),'np.shape(reoriented_cluster_normals)')
    # initial estimation of radius r
    no_pts                          = cluster_xyz.shape[0]
    t1 = np.sum(reoriented_cluster_xyz * reoriented_cluster_normals)
    t2 = np.sum(np.sum(reoriented_cluster_xyz, axis=0) * np.sum(reoriented_cluster_normals, axis=0))
    t3 = np.sum(reoriented_cluster_normals * reoriented_cluster_normals)
    t4 = np.sum(np.sum(reoriented_cluster_normals, axis=0) * np.sum(reoriented_cluster_normals, axis=0))
    r0 = -(no_pts*t1 - t2)/(no_pts*t3 - t4)
    #print(r0,'r0')
    # initial estimation of centre of the cylinder
    C0_projection = np.sum(reoriented_cluster_xyz + r0*reoriented_cluster_normals, axis = 0) / no_pts
    #print(C0_projection,'C0_projection')
    C0            = np.dot(R_inv, C0_projection)
    #print(C0,'C0')

    # ---------------------> initial estimation of closest point P on A0 nearest to the origin
    # Ref: https://math.stackexchange.com/questions/3158880/get-rectangular-coordinates-of-a-3d-point-with-the-polar-coordinates
    r0            = r0                                                    # initial estimation
    P0            = C0 - A0*(np.sum(C0 * A0))                             # initial estimation
    #print(C0,'C0')
    #print(A0,'A0')
    #print(P0,'P0')
    d0            = np.linalg.norm(P0)
    #print(d0,'d0')                                    # initial estimation
    N0            = P0 / d0
    # Ref: https://stackoverflow.com/questions/35749246/python-atan-or-atan2-what-should-i-use
    # use of arctan2 instead of arctan
    #print(N0,'N0')
    theta_0        = np.arctan2(np.sqrt(N0[0]*N0[0] + N0[1]*N0[1]), N0[2]) # initial estimation
    phi_0          = np.arctan2(N0[1], N0[0])                              # initial estimation
    #print(theta_0,'theta_0')
    #print(phi_0,'phi_0')
    N0_theta       = np.array([np.cos(phi_0)*np.cos(theta_0), np.sin(phi_0)*np.cos(theta_0), -np.sin(theta_0)])
    N0_phi         = np.array([-np.sin(phi_0), np.cos(phi_0), 0])
    cos_alpha_0    = A0[2]/N0_theta[2]
    sin_alpha_0    = (A0[0] - N0_theta[0]*A0[2]/N0_theta[2])/N0_phi[0]
    alpha_0        = np.arctan2(sin_alpha_0, cos_alpha_0)                  # initial estimation
    #print(alpha_0,'alpha_0')


    cyl_init_model = np.array([d0, theta_0, phi_0, alpha_0, r0]).reshape(5, 1)

    #print(cyl_init_model,'cyl_init_model')


    # Projecting points onto the cylinder's axis
    temp                 = np.sum((cluster_xyz - P0)*A0, axis = 1).reshape((no_pts, 1))
    projection           = P0 + temp.dot(A0.reshape((1, 3)))
    min_index            = np.argmin(projection[:, 0])
    max_index            = np.argmax(projection[:, 0])
    height               = np.linalg.norm(projection[min_index, :] - projection[max_index, :])
    directional_vector   = (projection[max_index, :] - projection[min_index, :])/np.linalg.norm(projection[max_index, :] - projection[min_index, :])
    center               = projection[min_index, :]  + directional_vector * height/2
    # --------> estimation of relative pose between cylinder's frame and camera's frame
    T = SE3(custom = {"t": center, "composed_rotational_components" : {"orders": ["z", "y", "z"], "angles" : [phi_0, theta_0, alpha_0]}}).T
    # initial estimation of residual
    cyl_error, cyl_quadratic_lost, rmse, _, _ \
                   = compute_distances_to_cylinder(cyl_init_model, cluster_xyz)
    result         = [cyl_init_model, cyl_quadratic_lost, cyl_error, T]
    return result

def fitting_cylinder(cluster_info, init_geometry_meta_info, optim_config_params, lambdified_jacobian_cylinder):
    # parsing
    internal_states = []
    cluster_xyz     = cluster_info[0]
    cluster_normals = cluster_info[1]
    no_pts          = cluster_xyz.shape[0]
    # computation of covariance feature

    # parsing
    damping_factor  = optim_config_params["damping_factor"]
    c               = optim_config_params["c"]
    threshold       = optim_config_params["threshold"]
    # parsing
    initial_model          = init_geometry_meta_info["model"]
    initial_error          = init_geometry_meta_info["error"]
    initial_quadratic_lost = init_geometry_meta_info["quadratic_lost"]
    # ------------------------> initial estimation of jacobian
    input_vector = np.append(cluster_xyz, np.ones((no_pts, 1)).dot(initial_model.transpose()), axis = 1)
    jacobian     = []
    for m in range(input_vector.shape[0]):
        jacobian.append(compute_jacobian_cylinder(lambdified_jacobian_cylinder, input_vector[m,:]))
    jacobian = np.array(jacobian)

    # ---------------------> update the initial value for refinement process
    error          = initial_error
    quadratic_lost = initial_quadratic_lost
    solution       = initial_model
    initial_state  = [solution[0,0],solution[1,0], solution[2,0], solution[3,0], solution[4,0], quadratic_lost]
    internal_states.append(initial_state)

    for x in range(100):
        # -----------> update the incremental change in all cylinder's parameters
        H = np.dot(jacobian.transpose(), jacobian)     # Hessian
        t1 = H + damping_factor * np.eye(H.shape[0])   # adding damping factor
        try:
            t2 = np.linalg.inv(t1)
        except:
            break
        t3 = np.dot(t2, jacobian.transpose())
        delta = -t3.dot(error)
        if np.linalg.norm(delta) <= threshold:
            break
        # -----------> update cylinder's parameters, error vector, quadratic lost and jacobian
        print('#######################')
        print(solution,'solution')
        print(delta,'delta')
        solution_temp        = solution + delta

        print(solution_temp,'solution_temp')

        error_temp, quadratic_lost_temp, rmse, _, _ \
                             = compute_distances_to_cylinder(solution_temp, cluster_xyz)

        internal_state      = [solution_temp[0,0],solution_temp[1,0], solution_temp[2,0], solution_temp[3,0], solution_temp[4,0], quadratic_lost_temp]
        internal_states.append(internal_state)

        print(quadratic_lost_temp,'quadratic_lost_temp')
        print(quadratic_lost,'quadratic_lost')

        if quadratic_lost_temp < quadratic_lost:
            # error decreased, reducing damping factor
            damping_factor   = damping_factor / c
            # update the solution
            solution         = solution_temp
            # update error vector
            error            = error_temp
            # update the quadratic lost
            quadratic_lost   = quadratic_lost_temp
            # update the jacobian
            input_vector     = np.append(cluster_xyz, np.ones((no_pts, 1)).dot(solution.transpose()), axis = 1)
            jacobian     = []
            for m in range(input_vector.shape[0]):
                jacobian.append(compute_jacobian_cylinder(lambdified_jacobian_cylinder, input_vector[m,:]))
            jacobian = np.array(jacobian)
        else:
            # error increased: discard and raising damping factor
            damping_factor   = c * damping_factor

    d              = solution[0, 0]
    theta          = solution[1, 0]
    phi            = solution[2, 0]
    alpha          = solution[3, 0]
    r              = solution[4, 0]
    N              = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
    N_theta        = np.array([np.cos(phi)*np.cos(theta), np.sin(phi)*np.cos(theta), -np.sin(theta)])
    N_phi          = np.array([-np.sin(phi), np.cos(phi), 0])
    P              = d*N
    A              = N_theta * np.cos(alpha) + N_phi * np.sin(alpha)
    print(A,'A')
    exit()
    # Projecting points onto the cylinder's axis
    temp                 = np.sum((cluster_xyz - P)*A, axis = 1).reshape((no_pts, 1))
    projection           = P + temp.dot(A.reshape((1, 3)))
    print(projection,'projection')
    min_index            = np.argmin(projection[:, 0])
    max_index            = np.argmax(projection[:, 0])
    height               = np.linalg.norm(projection[min_index, :] - projection[max_index, :])
    directional_vector   = (projection[max_index, :] - projection[min_index, :])/np.linalg.norm(projection[max_index, :] - projection[min_index, :])
    center               = projection[min_index, :]  + directional_vector * height/2
    # --------> estimation of relative pose between cylinder's frame and camera's frame
    print(np.shape(center),'center')
    print(center,'center')
    print(np.shape(phi),'phi')
    print(phi,'phi')
    print(np.shape(theta),'theta')
    print(theta,'theta')
    print(np.shape(alpha),'alpha')
    print(alpha,'alpha')



    T = SE3(custom = {"t": center, "composed_rotational_components" : {"orders": ["z", "y", "z"], "angles" : [phi, theta, alpha]}}).T
    # --------> summarize
    result   = [[solution, quadratic_lost, T, height, A], internal_states]
    return result

if __name__ == "__main__":
    # initialization
    vis             = True
    initial_model   = []
    optimized_model = []
    lambdified_jacobian_cylinder = jacobian_cylinder()

    # load point cloud data (path to point cloud data)
    pcd = o3d.io.read_point_cloud("camera/pcddata_cylinder.pcd")
    # estimate normal vectors
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))

    # get cluster_info
    xyz     = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    cluster_info = [xyz, normals]

    print(xyz,'xyz')
    print(normals,'normals')
    exit()
    # optimizer configuration parameters
    optim_config_params                   = {}
    optim_config_params["damping_factor"] = 100
    optim_config_params["c"]              = 2
    optim_config_params["threshold"]      = 10e-6
    # dictionary to collect geometric relevant meta-info
    geometry_meta_info                    = {}
    geometry_meta_info["init"]            = {}
    geometry_meta_info["refined"]         = {}
    # ----------------> initiating sphere model
    init_result = initiating_cylinder(cluster_info)
    # init_result[0] = np.array([[0.5157442827161066 ], [0.4815114070242191], [0.5269788220369058]\
    #                            , [2.952551266794164], [-0.03533251082674181]])

    geometry_meta_info["init"]["model"]          =  init_result[0]
    #(init_result[0],'init_result[0]')
    geometry_meta_info["init"]["quadratic_lost"] =  init_result[1]
    geometry_meta_info["init"]["error"]          =  init_result[2]
    geometry_meta_info["init"]["T"]              =  init_result[3]
    # ----------------> optimizing sphere model
    refined_result = fitting_cylinder(cluster_info, geometry_meta_info["init"], optim_config_params, lambdified_jacobian_cylinder)
    geometry_meta_info["refined"]["model"]          = refined_result[0][0]
    # geometry_meta_info["refined"]["model"]          = np.array([[0.5157442827161066 ], [0.4815114070242191], [0.5269788220369058]\
    #                            , [2.952551266794164], [-0.03533251082674181]])
    print(refined_result[0][0],'refined_result[0][0]')

    geometry_meta_info["refined"]["quadratic_lost"] = refined_result[0][1]
    geometry_meta_info["refined"]["T"]              = refined_result[0][2]
    geometry_meta_info["refined"]["height"]         = refined_result[0][3]
    quadratic_lost_evolution                        = []
    for i in range(len(refined_result[1])):
        quadratic_lost_evolution.append(refined_result[1][i][5])
    plt.plot(quadratic_lost_evolution)
    plt.show()
    if vis == True:
        models = []
        r      = geometry_meta_info["refined"]["model"][4]
        height = geometry_meta_info["refined"]["height"]
        T      = geometry_meta_info["refined"]["T"]



        mesh_model  = o3d.geometry.TriangleMesh.create_cylinder(radius = abs(r), height = height, resolution = 100)
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.07, origin=[0, 0, 0])


        mesh_model.compute_vertex_normals()
        mesh_model.paint_uniform_color([0, 0.51, 0.14])
        # rotation matrix that align Z-axis to X-axis
        R = SO3(ypr = np.array([0, np.pi/2, 0])).R
        mesh_model.rotate(R, center=(0, 0, 0))
        world_frame.rotate(R, center=(0, 0, 0))
        # CAUTION: # the cylinder and the world frame would arbitrarily oriented in both direction
        # However, this is not perceptually important since the cylinder is symmetric geometry
        mesh_model.transform(T)
        world_frame.transform(T)
        mesh_model = mesh_model + world_frame
        models.append(mesh_model)
        models[0] = models[0].transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        models.append(pcd)
        o3d.visualization.draw_geometries(models, point_show_normal=True)
