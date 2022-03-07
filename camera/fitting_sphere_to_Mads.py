import sympy as sp
import numpy as np
import open3d as o3d
from sympy.utilities.autowrap import autowrap
import matplotlib.pyplot as plt
import time
print("Environment Ready")
# ------------------------------------>
def jacobian_sphere():
    # Notation:
    # M_i = (m_ix, m_iy, m_iz) : range point M_i
    # C   = (c_x, c_y, c_z)    : sphere centre
    # r   : sphere radius
    m_ix, m_iy, m_iz = sp.symbols('m_ix m_iy m_iz')
    c_x, c_y, c_z    = sp.symbols('c_x c_y c_z')
    r                = sp.symbols('r')
    # distance of element M_i from estimated surface
    e_sph     = sp.sqrt((m_ix - c_x)**2 + (m_iy - c_y)**2 + (m_iz - c_z)**2) - sp.sign(r)*r
    # jacobian of distance function (i.e. lost function)
    J_cx      = sp.diff(e_sph, c_x)
    J_cy      = sp.diff(e_sph, c_y)
    J_cz      = sp.diff(e_sph, c_z)
    J_r       = sp.diff(e_sph, r)
    jacobian_sphere  = sp.Matrix([J_cx, J_cy, J_cz, J_r])
    jacobian_sphere  = jacobian_sphere.subs(sp.diff(sp.sign(r), r), 0)
    jacobian_sphere  = autowrap(jacobian_sphere, backend="cython", args = [m_ix, m_iy, m_iz, c_x, c_y, c_z, r])
    return jacobian_sphere


def compute_jacobian_sphere(lambdified_jacobian_sphere, input_vector):
    result  = lambdified_jacobian_sphere(*input_vector).reshape((4,)).astype('float32')
    return result

def compute_distances_to_sphere(model, cluster_xyz):
    no_pts           = cluster_xyz.shape[0]
    signed_distances = (np.linalg.norm(cluster_xyz - model[:3, 0], axis=1) - abs(model[3, 0])).reshape((no_pts, 1))
    lost             = np.sum(signed_distances * signed_distances) / no_pts
    rmse             = np.sqrt(lost)
    return signed_distances, lost, rmse

def initiating_sphere(cluster_info):
    cluster_xyz                 = cluster_info[0]
    cluster_normals             = cluster_info[1]
    no_pts                      = cluster_xyz.shape[0]
    t1                          = np.sum(cluster_xyz * cluster_normals)
    t2                          = np.sum(np.sum(cluster_xyz, axis=0) * np.sum(cluster_normals, axis=0))
    t3                          = np.sum(cluster_normals * cluster_normals)
    t4                          = np.sum(np.sum(cluster_normals, axis=0) * np.sum(cluster_normals, axis=0))
    r                           = -(no_pts*t1 - t2)/(no_pts*t3 - t4)
    C                           = np.sum(cluster_xyz + r*cluster_normals, axis = 0) / no_pts
    sphere_model                = np.array([C[0], C[1], C[2], r]).reshape(4, 1)
    T                           = np.eye(4)
    T[:3, 3]                    = sphere_model[:3, 0]
    sphere_error, sphere_quadratic_lost, rmse = compute_distances_to_sphere(sphere_model, cluster_xyz)
    result                      = [sphere_model, sphere_quadratic_lost, sphere_error, T]
    return result

def fitting_sphere(cluster_info, init_geometry_meta_info, optim_config_params, lambdified_jacobian_sphere):
    # parsing
    internal_states        = []
    cluster_xyz            = cluster_info[0]
    cluster_normals        = cluster_info[1]
    no_pts                 = cluster_xyz.shape[0]
    # parsing
    damping_factor         = optim_config_params["damping_factor"]
    c                      = optim_config_params["c"]
    threshold              = optim_config_params["threshold"]
    # parsing
    initial_model          = init_geometry_meta_info["model"]
    initial_error          = init_geometry_meta_info["error"]
    initial_quadratic_lost = init_geometry_meta_info["quadratic_lost"]
    # ------------------------> initial estimation of jacobian
    input_vector = np.append(cluster_xyz, np.ones((no_pts, 1)).dot(initial_model.transpose()), axis = 1)
    jacobian     = []
    start_timer = time.perf_counter()
    for m in range(input_vector.shape[0]):
        jacobian.append(compute_jacobian_sphere(lambdified_jacobian_sphere, input_vector[m, :]))
    jacobian = np.array(jacobian)

    end_timer = time.perf_counter()

    final_time = end_timer - start_timer
    print(final_time,'final_time')

    exit()

    # -------------------> update initial values for refinement process
    quadratic_lost = initial_quadratic_lost
    error          = initial_error
    solution       = initial_model
    initial_state  = [solution[0,0],solution[1,0], solution[2,0], solution[3,0], quadratic_lost]
    internal_states.append(initial_state)

    for x in range(100):

        # -------> update incremental change in estimation of sphere's centre and radius
        H        = np.dot(jacobian.transpose(), jacobian) # Hesian
        t1       = H + damping_factor * np.eye(H.shape[0]) # adding damping factor
        try:
            t2       = np.linalg.inv(t1)
        except:
            break
        t3       = np.dot(t2, jacobian.transpose())
        delta    = -t3.dot(error)
        if np.linalg.norm(delta) <= threshold:
            break

        # -------> update sphere centre and radius
        solution_temp       = solution + delta
        # -------> update error vector and quadratic lost and jacobian

        error_temp, quadratic_lost_temp, rmse_temp\
                              = compute_distances_to_sphere(solution_temp, cluster_xyz)

        internal_state      = [solution_temp[0,0],solution_temp[1,0], solution_temp[2,0], solution_temp[3,0], quadratic_lost_temp]
        internal_states.append(internal_state)
        # Since we do not know if the current incremental change in solution resulting in reduced quadratic loss
        if quadratic_lost_temp < quadratic_lost:
            # error decreased, reducing damping factor
            print('#############################################')
            print(x,'count')
            print(quadratic_lost_temp,'quadratic_lost_temp')
            print(quadratic_lost,'quadratic_lost')

            damping_factor = damping_factor / c
            # update the solution
            print(solution,'solution')
            print(solution_temp,'solution_temp')
            solution       = solution_temp
            # update error vector
            error          = error_temp
            # update the quadratic lost
            quadratic_lost = quadratic_lost_temp
            # update jacobian
            input_vector   = np.append(cluster_xyz, np.ones((no_pts, 1)).dot(solution.transpose()), axis = 1)
            jacobian       = []
            for m in range(input_vector.shape[0]):
                jacobian.append(compute_jacobian_sphere(lambdified_jacobian_sphere, input_vector[m, :]))
            jacobian = np.array(jacobian)
            print(jacobian,'jacobian')
        else:
            # error increased: discard and raising damping factor
            damping_factor = c * damping_factor

    # ---------> transformation that aligns sphere's frame and camera's frame
    T = np.eye(4)
    T[:3, 3] = solution[:3, 0]
    # ---------> summarize
    result   = [[solution, quadratic_lost, T], internal_states]
    return result

if __name__ == "__main__":
    # initialization
    vis             = True
    initial_model   = []
    optimized_model = []
    lambdified_jacobian_sphere = jacobian_sphere()

    # load point cloud data (path to point cloud data)
    pcd = o3d.io.read_point_cloud("camera/pcddata.pcd")
    # estimate normal vectors
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))

    # get cluster_info
    xyz     = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    cluster_info = [xyz, normals]

    # optimizer configuration parameters
    optim_config_params                   = {}
    optim_config_params["damping_factor"] = 10000
    optim_config_params["c"]              = 2
    optim_config_params["threshold"]      = 10e-7
    # dictionary to collect geometric relevant meta-info
    geometry_meta_info                    = {}
    geometry_meta_info["init"]            = {}
    geometry_meta_info["refined"]         = {}
    # ----------------> initiating sphere model
    init_result = initiating_sphere(cluster_info)
    geometry_meta_info["init"]["model"]          =  init_result[0]
    geometry_meta_info["init"]["quadratic_lost"] =  init_result[1]
    geometry_meta_info["init"]["error"]          =  init_result[2]
    geometry_meta_info["init"]["T"]              =  init_result[3]
    # ----------------> optimizing sphere model
    refined_result = fitting_sphere(cluster_info, geometry_meta_info["init"], optim_config_params, lambdified_jacobian_sphere)
    geometry_meta_info["refined"]["model"]          = refined_result[0][0]
    geometry_meta_info["refined"]["quadratic_lost"] = refined_result[0][1]
    geometry_meta_info["refined"]["T"]              = refined_result[0][2]
    quadratic_lost_evolution                        = []
    for i in range(len(refined_result[1])):
        quadratic_lost_evolution.append(refined_result[1][i][4])
    plt.plot(quadratic_lost_evolution)
    plt.show()
    if vis == True:
        models = []
        r = geometry_meta_info["refined"]["model"][3]
        T = geometry_meta_info["refined"]["T"]
        mesh_model  = o3d.geometry.TriangleMesh.create_sphere(radius = abs(r))
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.07, origin=[0, 0, 0])
        mesh_model.compute_vertex_normals()
        mesh_model.paint_uniform_color([0, 0, 1])
        mesh_model.transform(T)
        world_frame.transform(T)
        mesh_model = mesh_model + world_frame
        models.append(mesh_model)
        models[0] = models[0].transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        models.append(pcd)
        o3d.visualization.draw_geometries(models, point_show_normal=True)
