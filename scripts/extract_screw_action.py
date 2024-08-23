import math
import pickle
import random

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm, expm
from scipy.optimize import curve_fit
from argparse import ArgumentParser

REVOLTE3D_SEARCH_ITERATIONS = 100

def config_parser(parser=None):
    if parser is None:
        parser = ArgumentParser("ScrewMimic")
    parser.add_argument('--folder_name', type=str)
    parser.add_argument('--hand', type=str, default='left')
    parser.add_argument('--use_camera_frame', action='store_true', default=False)
    parser.add_argument('--use_pos_only', action='store_true', default=False, help='Use position or (position + orientation) to compare original and predicted trajectories')
    parser.add_argument('--show_revolute_axis', action='store_true', default=False, help='Show the calculated revolute axis in matplotlib')
    parser.add_argument('--show_prismatic_axis', action='store_true', default=False, help='Show the calculated prismatic axis in matplotlib')
    parser.add_argument('--show_revolute3d_axis', action='store_true', default=False, help='Show the calculated revolute3d axis in matplotlib')
    return parser

def compute_screw_axis_revolute(start_T, final_T):
    # delta_T = np.dot(final_T, np.linalg.inv(start_T))
    delta_T = np.dot(final_T, np.linalg.inv(start_T))

    # Compute the matrix logarithm
    log_T = logm(delta_T)

    # Extract linear velocities
    linear_velocities = log_T[:3, 3]

    # Extract skew-symmetric matrix (angular velocities)
    S = log_T[:3, :3]
    # print("S: ", S)

    # Calculate angular velocities from the skew-symmetric matrix
    angular_velocities = np.array([S[2, 1] - S[1, 2], S[0, 2] - S[2, 0], S[1, 0] - S[0, 1]]) / 2.0
    # print("angular_velocities: ", angular_velocities)

    # Combine linear and angular velocities to get the twist
    twist = np.concatenate((linear_velocities, angular_velocities))
    # print("Twist vector:", twist)

    screw_axis = angular_velocities / np.linalg.norm(angular_velocities)
    theta = np.linalg.norm(angular_velocities)
    q = np.cross(screw_axis, linear_velocities) / theta
    # print("screw_axis: ", screw_axis)
    # print("q: ", q)
    
    return screw_axis, q

def circle_equation(x, h, k, l, r):
    return (x[0] - h)**2 + (x[1] - k)**2 + (x[2] - l)**2 - r**2

def compute_screw_axis_revolute3d(Ts, pts_to_compute_normal=[10, 1, 12, 3]):
    # Example 3D points
    data_points =  []
    for T in Ts:
        pos = np.array(T)[:3, 3]
        data_points.append(pos)
    data_points = np.array(data_points)

    initial_guess = [0, 0, 0, 1]
    # Perform least squares fit
    params, covariance = curve_fit(circle_equation, data_points.T, np.zeros(data_points.shape[0]), p0=initial_guess)
    arc_centre_x, arc_centre_y, arc_centre_z, radius = params
    q = np.array([arc_centre_x, arc_centre_y, arc_centre_z])
    # print(f"Center of the circle: ({arc_centre_x}, {arc_centre_y}, {arc_centre_z})")
    # print(f"Radius of the circle: {radius}")

    pt1, pt2, pt3, pt4 = pts_to_compute_normal
    temp_vec_1 = data_points[pt1] - data_points[pt2] 
    temp_vec_2 = data_points[pt3] - data_points[pt4]
    normal_vector = np.cross(temp_vec_1, temp_vec_2)

    # Normalize the normal vector to get the axis of rotation
    axis_of_rotation = normal_vector / np.linalg.norm(normal_vector)
    # print(f"Axis of rotation: {axis_of_rotation}")

    return axis_of_rotation, q

def compute_screw_axis_prismatic(Ts):
    # Fit a line
    data_points =  []
    for T in Ts:
        pos = np.array(T)[:3, 3]
        data_points.append(pos)
    data_points = np.array(data_points)
    datamean = data_points.mean(axis=0)
    uu, dd, vv = np.linalg.svd(data_points - datamean)
    s = vv[0]
    s_hat = s / np.linalg.norm(s)

    start_T = Ts[0]
    q = np.array([start_T[0, 3], start_T[1, 3], start_T[2, 3]])

    return s_hat, q

def compute_trajectory_prismatic(T0, s_hat, len_hand_pts, theta_step=0.007):
    computed_Ts = []
    computed_Ts.append(T0)
    twist = [0, 0, 0] + s_hat.tolist()
    w = twist[:3]
    w_matrix = [
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0],
    ]

    S = [
        [w_matrix[0][0], w_matrix[0][1], w_matrix[0][2], twist[3]],
        [w_matrix[1][0], w_matrix[1][1], w_matrix[1][2], twist[4]],
        [w_matrix[2][0], w_matrix[2][1], w_matrix[2][2], twist[5]],
        [0, 0, 0, 0]
    ]

    # calculate the thetas
    thetas = []
    for i in range(1, len_hand_pts):
        thetas.append(i*theta_step)

    for theta in thetas:
        S_theta = theta * np.array(S)
        T1 = np.dot(expm(S_theta), T0)
        computed_Ts.append(T1)
    return computed_Ts

def compute_trajectory_screw(T0, s_hat, q, len_hand_pts, theta_step=0.0):
    computed_Ts = []
    w = s_hat
    v = -np.cross(s_hat, q)
    twist = np.concatenate((w,v)) 
    # Calculate the matrix form of the twist vector
    w = twist[:3]
    w_matrix = [
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0],
    ]
    # print("w_matrix: ", w_matrix)
    S = [
        [w_matrix[0][0], w_matrix[0][1], w_matrix[0][2], twist[3]],
        [w_matrix[1][0], w_matrix[1][1], w_matrix[1][2], twist[4]],
        [w_matrix[2][0], w_matrix[2][1], w_matrix[2][2], twist[5]],
        [0, 0, 0, 0]
    ]
    computed_Ts.append(T0)

    # calculate the thetas
    thetas = []
    for i in range(1, len_hand_pts):
        thetas.append(i*theta_step)
    
    # Calculate the transformation of the point when moved by theta along the screw axis
    for theta in thetas:
        S_theta = theta * np.array(S)
        T1 = np.dot(expm(S_theta), T0)
        computed_Ts.append(T1)
    
    return computed_Ts

def compute_trajectory_revolute3d(T0, axis, q, len_hand_pts, theta_step=0.05):
    computed_Ts = []
    original_orientation = np.array(T0)[:3,:3]
    h = 0 # pure rotation
    s_hat = axis
    w = s_hat
    v = -np.cross(s_hat, q)
    twist = np.concatenate((w,v)) 
    # Calculate the matrix form of the twist vector
    w = twist[:3]
    w_matrix = [
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0],
    ]
    # print("w_matrix: ", w_matrix)
    S = [
        [w_matrix[0][0], w_matrix[0][1], w_matrix[0][2], twist[3]],
        [w_matrix[1][0], w_matrix[1][1], w_matrix[1][2], twist[4]],
        [w_matrix[2][0], w_matrix[2][1], w_matrix[2][2], twist[5]],
        [0, 0, 0, 0]
    ]
    computed_Ts.append(T0)

    # calculate the thetas
    thetas = []
    for i in range(1, len_hand_pts):
        thetas.append(i*theta_step)
    
    # Calculate the transformation of the point when moved by theta along the screw axis
    for theta in thetas:
        S_theta = theta * np.array(S)
        T1 = np.dot(expm(S_theta), T0)
        T1 = [
            [original_orientation[0][0], original_orientation[0][1], original_orientation[0][2], T1[0, 3]],
            [original_orientation[1][0], original_orientation[1][1], original_orientation[1][2], T1[1, 3]],
            [original_orientation[2][0], original_orientation[2][1], original_orientation[2][2], T1[2, 3]],
            [0, 0, 0, 1]
        ]
        computed_Ts.append(T1)
    
    return computed_Ts

def compute_trajectory_score(original_Ts, computed_Ts, log=False):
    original_Ts = np.array(original_Ts)
    computed_Ts = np.array(computed_Ts)
    trajectory_pos_dist = 0
    trajectory_orn_dist = 0
    for i in range(len(original_Ts)):
        original_pos = original_Ts[i][:3, 3]
        computed_pos = computed_Ts[i][:3, 3]
        point_dist = np.linalg.norm(original_pos-computed_pos)
        if log:
            print(f"index {i}: ", original_pos, computed_pos, point_dist)
        trajectory_pos_dist += point_dist

        original_rot = original_Ts[i][:3, :3]
        computed_rot = computed_Ts[i][:3, :3]
        original_quat = R.from_matrix(original_rot).as_quat()
        original_quat = original_quat / np.linalg.norm(original_quat)
        computed_quat = R.from_matrix(computed_rot).as_quat()
        computed_quat = computed_quat / np.linalg.norm(computed_quat)
        orn_dist = 1 - np.dot(original_quat, computed_quat)**2          # Followed https://math.stackexchange.com/questions/90081/quaternion-distance
        trajectory_orn_dist += orn_dist

    return trajectory_pos_dist, trajectory_orn_dist 

def main():
    args = config_parser().parse_args()
    random.seed(1)
    np.random.seed(1)
    path = f'data/{args.folder_name}/hand_poses.pickle'

    with open(path, 'rb') as handle:
        hand_dict = pickle.load(handle)

    # Choose the acting hand
    hand_dict = hand_dict[args.hand]
    print("hand_dict: ", hand_dict.keys())

    # If you want the screw axis to be in the robot base frame or left hand frame, you need the extrinsics 
    if not args.use_camera_frame:
        with open(f'data/{args.folder_name}/extrinsic.pickle', 'rb') as handle:
            extr = pickle.load(handle)
    # else the screw axis is computed w.r.t camera frame
    else:
        extr = np.eye(4)

    # Create a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # ------------------- Get the 6-DoF hand poses and the delta poses -----------------------
    hand = hand_dict[next(iter(hand_dict))]
    init_pos = hand[f'{args.hand}_hand_pos']
    init_pos /= 1000
    init_orn = hand[f'{args.hand}_hand_orn']
    init_orn_matrix = R.from_rotvec(np.array(init_orn)).as_matrix()
    T0 = [
            [init_orn_matrix[0][0], init_orn_matrix[0][1], init_orn_matrix[0][2], init_pos[0]],
            [init_orn_matrix[1][0], init_orn_matrix[1][1], init_orn_matrix[1][2], init_pos[1]],
            [init_orn_matrix[2][0], init_orn_matrix[2][1], init_orn_matrix[2][2], init_pos[2]],
            [0, 0, 0, 1]
        ]
    T0_world = np.dot(extr, T0)
    first_hand_pose = T0_world.copy()
    delta_Ts = []
    Ts = []
    Ts.append(T0_world)
    for i, h in enumerate(hand_dict.keys()):
        if i == 0:
            continue
        hand_pos = hand_dict[h][f'{args.hand}_hand_pos']
        hand_pos /= 1000 # convert positions into meters
        hand_orn = hand_dict[h][f'{args.hand}_hand_orn']
        hand_orn_matrix = R.from_rotvec(np.array(hand_orn)).as_matrix()

        T1 = [
            [hand_orn_matrix[0][0], hand_orn_matrix[0][1], hand_orn_matrix[0][2], hand_pos[0]],
            [hand_orn_matrix[1][0], hand_orn_matrix[1][1], hand_orn_matrix[1][2], hand_pos[1]],
            [hand_orn_matrix[2][0], hand_orn_matrix[2][1], hand_orn_matrix[2][2], hand_pos[2]],
            [0, 0, 0, 1]
        ]
        T1 = np.array(T1)
        T1_world = np.dot(extr, T1)
        Ts.append(T1_world)

        # visualizing the hand positions and orientations
        ax.scatter(T1_world[0, 3], T1_world[1, 3], T1_world[2, 3], c=([[0.5,0.5,0.5]]), marker='o')
    # ----------------------------------------------------------------------

    len_hand_pts = len(Ts)
    
    # Screw joint type 1: revolute joint
    min_screw_revolute_dist = 1000000
    final_axis_screw_revolute = None
    final_q_screw_revolute = None
    final_computed_Ts_screw = None
    final_start_T_revolute, final_end_T_revolute = None, None
    final_idx_revolute = None
    for i in range(len(Ts)):
        for j in range(i+1, len(Ts)):
            # Compute the screw axis based on 2 poses 
            start_T = Ts[i]
            final_T = Ts[j]
            axis, q = compute_screw_axis_revolute(start_T, final_T)

            T0 = first_hand_pose
            # Compute the trajectory based on the previously obtained revolute screw axis 
            computed_Ts_screw = compute_trajectory_screw(T0, axis, q, len_hand_pts, theta_step=0.06)    # This bit is a bit hacky. theta_step depends on the rate at which you sample the 
                                                                                                        # human video and obtain hand poses or how fast the human is doing the motion 
                                                                                                        # Make sure to modify theta_step for your videos
            screw_pos_dist, screw_orn_dist,  = compute_trajectory_score(Ts, computed_Ts_screw)
            # print("revolute axis: ", axis, q, screw_pos_dist)
            if args.use_pos_only:
                screw_dist = screw_pos_dist
            else:
                screw_dist = screw_pos_dist + screw_orn_dist

            # print("pos_dist, orn_dist: ", screw_pos_dist, screw_orn_dist)

            # print("screw_pos_dist ", i, j, screw_dist)
            if screw_dist < min_screw_revolute_dist:
                min_screw_revolute_dist = screw_dist
                final_axis_screw_revolute = axis
                final_q_screw_revolute = q
                final_computed_Ts_screw_revolute = computed_Ts_screw
                final_start_T_revolute = start_T
                # final_end_T_revolute = final_T
                # final_idx_revolute = (i, j)
    print("Revolute: s_hat, q, min_screw_revolute_dist: ", final_axis_screw_revolute, final_q_screw_revolute, min_screw_revolute_dist)
    
    # Screw joint type 2: revolute3d joint
    min_screw_revolute3d_dist = 10000000
    for i in range(REVOLTE3D_SEARCH_ITERATIONS):
        idxs = random.sample(range(0, len(Ts)), 4)
        axis, q = compute_screw_axis_revolute3d(Ts, pts_to_compute_normal=idxs)

        T0 = first_hand_pose
        # Compute the trajectory based on the previously obtained screw axis 
        computed_Ts_screw = compute_trajectory_revolute3d(T0, axis, q, len_hand_pts, theta_step=0.3) # Make sure to modify theta_step for your videos
        screw_pos_dist, screw_orn_dist = compute_trajectory_score(Ts, computed_Ts_screw)
        if args.use_pos_only:
            screw_dist = screw_pos_dist
        else:
            screw_dist = screw_pos_dist + screw_orn_dist

        if screw_dist < min_screw_revolute3d_dist:
            min_screw_revolute3d_dist = screw_dist
            final_axis_screw_revolute3d = axis
            final_q_screw_revolute3d = q
            final_computed_Ts_screw_revolute3d = computed_Ts_screw
            # final_idxs = idxs
    print("Revolute3D: s_hat, q, min_screw_revolute_dist: ", final_axis_screw_revolute3d, final_q_screw_revolute3d, min_screw_revolute3d_dist)

    # Screw joint type 3: prismatic joint
    final_axis_screw_prismatic, final_q_screw_prismatic = compute_screw_axis_prismatic(Ts)
    T0 = first_hand_pose
    final_computed_Ts_screw_prismatic = compute_trajectory_prismatic(T0, final_axis_screw_prismatic, len_hand_pts, theta_step=0.007) # Make sure to modify theta_step for your videos
    screw_pos_dist, screw_orn_dist = compute_trajectory_score(Ts, final_computed_Ts_screw_prismatic)
    if args.use_pos_only:
        min_screw_prismatic_dist = screw_pos_dist
    else:
        min_screw_prismatic_dist = screw_pos_dist + screw_orn_dist
    print("Prismatic: s_hat, q, min_screw_revolute_dist: ", final_axis_screw_prismatic, final_q_screw_prismatic, min_screw_prismatic_dist)

    # Obtain the screw type and the final screw axis
    if min_screw_revolute_dist < min_screw_prismatic_dist and min_screw_revolute_dist < min_screw_revolute3d_dist:
        final_s_hat = final_axis_screw_revolute
        final_q = final_q_screw_revolute
        screw_type = 'revolute'
    elif min_screw_prismatic_dist < min_screw_revolute_dist and min_screw_prismatic_dist < min_screw_revolute3d_dist:
        final_s_hat = final_axis_screw_prismatic
        final_q = final_q_screw_prismatic
        screw_type = 'prismatic'
    elif min_screw_revolute3d_dist < min_screw_prismatic_dist and min_screw_revolute3d_dist < min_screw_revolute_dist:
        final_s_hat = final_axis_screw_revolute3d
        final_q = final_q_screw_revolute3d
        screw_type = 'revolute3d'
    print("screw_type, final_s_hat, final_q: ", screw_type, final_s_hat, final_q)

    # # Optional: Get the q closest to the centroid of the object's point cloud
    # data = np.loadtxt('perception_model_data/bottle/bottle_4.csv', delimiter=',').astype(np.float32) # obtain point cloud of the object
    # points = data[:, :3]
    # centroid = np.mean(points, axis=0)
    # w = centroid - final_q_screw
    # t = np.dot(w, final_axis_screw) / np.dot(final_axis_screw, final_axis_screw)
    # final_q_screw = final_q_screw + t * final_axis_screw
    
    # save_dict = {
    #     's_hat': final_s_hat,
    #     'q': final_q
    # }
    # print("save_dict: ", save_dict)

    # Visualize the three kinds of screw axes
    if args.show_revolute_axis:
        length = 1
        endpoint_1 = final_q_screw_revolute + length * final_axis_screw_revolute
        endpoint_2 = final_q_screw_revolute - length * final_axis_screw_revolute
        ax.scatter(final_start_T_revolute[0][3], final_start_T_revolute[1][3], final_start_T_revolute[2][3], color=[0,0,0], marker='o', s=144)
        # ax.scatter(final_end_T[0][3], final_end_T[1][3], final_end_T[2][3], color=[0,1,1], marker='o',s=144)
        # ax.scatter(final_q_screw_revolute[0], final_q_screw_revolute[1], final_q_screw_revolute[2], color=[0,0.2,0.2], marker='o',s=144)
        ax.plot([endpoint_2[0], endpoint_1[0]], [endpoint_2[1], endpoint_1[1]], [endpoint_2[2], endpoint_1[2]], c=[0,0,1])
        # for pose in final_computed_Ts_screw_revolute:
        #     ax.scatter(pose[0][3], pose[1][3], pose[2][3], color='b', marker='o')
    if args.show_revolute3d_axis:
        length = 1
        endpoint_1 = np.array([final_q_screw_revolute3d[0], final_q_screw_revolute3d[1], final_q_screw_revolute3d[2]]) + length * final_axis_screw_revolute3d
        endpoint_2 = np.array([final_q_screw_revolute3d[0], final_q_screw_revolute3d[1], final_q_screw_revolute3d[2]]) - length * final_axis_screw_revolute3d
        ax.scatter(final_q_screw_revolute3d[0], final_q_screw_revolute3d[1], final_q_screw_revolute3d[2], color='r', marker='o')
        ax.plot([endpoint_2[0], endpoint_1[0]], [endpoint_2[1], endpoint_1[1]], [endpoint_2[2], endpoint_1[2]], c=[1,0,1])
        # for pose in final_computed_Ts_screw_revolute3d:
        #     ax.scatter(pose[0][3], pose[1][3], pose[2][3], color=[1,0,1], marker='o')
    if args.show_prismatic_axis:
        length = 1
        endpoint_1 = np.array([final_q_screw_prismatic[0], final_q_screw_prismatic[1], final_q_screw_prismatic[2]]) + length * final_axis_screw_prismatic
        endpoint_2 = np.array([final_q_screw_prismatic[0], final_q_screw_prismatic[1], final_q_screw_prismatic[2]]) - length * final_axis_screw_prismatic
        ax.scatter(final_q_screw_prismatic[0], final_q_screw_prismatic[1], final_q_screw_prismatic[2], color='r', marker='o')
        ax.plot([endpoint_2[0], endpoint_1[0]], [endpoint_2[1], endpoint_1[1]], [endpoint_2[2], endpoint_1[2]], c=[1,1,0])
        for pose in final_computed_Ts_screw_prismatic:
            ax.scatter(pose[0][3], pose[1][3], pose[2][3], color=[1,0,1], marker='o')


    # Set labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')       
    ax.set_xlim([0.3, 0.9])
    ax.set_ylim([-0.2, 0.2])
    ax.set_zlim([0.5, 1])
    plt.show() 


if __name__ == "__main__":
    main()