import os
import pickle
import cv2

import numpy as np

from argparse import ArgumentParser
from scipy.spatial.transform import Rotation as R



def get_XYZ(depth_Z, pix_x, pix_y):
    # Intrinsic matrix for Realsense camera
    # intr = np.array([
    #         [606.76220703,   0,         308.31533813],
    #         [  0,         606.91583252, 255.4833374 ],
    #         [  0,           0,           1        ]
    #     ])

    # Intrinsic matrix for Tiago's camera:
    intr = np.array([
        [523.9963414139355, 0.0, 328.83202929614686],
        [0.0, 524.4907272320442, 237.83703502879925],
        [0.0, 0.0, 1.0]
    ])

    click_z = depth_Z
    click_x = (pix_x-intr[0, 2]) * \
        click_z/intr[0, 0]
    click_y = (pix_y-intr[1, 2]) * \
        click_z/intr[1, 1]

    # 3d point in camera coordinates
    point_3d = np.asarray([click_x, click_y, click_z])
    return point_3d


def find_non_zero_depth(depth_img, left_wrist_xy):
    directions = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
    depth_value = depth_img[left_wrist_xy[1],left_wrist_xy[0]]
    if depth_value != 0:
        return depth_value
    for dist in range(1,7):
        for dir in directions:
            new_pix = left_wrist_xy + dist * dir
            depth_value = depth_img[new_pix[1],new_pix[0]]
            if depth_value != 0:
                return depth_value

def config_parser(parser=None):
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument('--folder_name', type=str)
    return parser

def get_hand_info(img_original_bgr, depth_img, pred_output_list, image_path, hand_info_dict):
    if len(pred_output_list['left_hand'].keys()) > 0:
        left_hand_orn = np.array(pred_output_list['left_hand']['pred_hand_pose'])[0][:3]
        left_wrist_xy = np.array(pred_output_list['left_hand']['pred_joints_img'][0][:2])
        left_wrist_xy = np.array([round(x) for x in left_wrist_xy])

        if left_wrist_xy[1] < depth_img.shape[0] and left_wrist_xy[0] < depth_img.shape[1]:
            depth_val = find_non_zero_depth(depth_img, left_wrist_xy)
            if depth_val is not None:
                print("depth value: ", depth_img[left_wrist_xy[1],left_wrist_xy[0]], depth_val)
                print("left_wrist_xy: ", left_wrist_xy)
                left_hand_pos = get_XYZ(depth_val, left_wrist_xy[0], left_wrist_xy[1])
                hand_info_dict["left"][image_path] = {
                    'color_img': img_original_bgr,
                    'depth_img': depth_img,
                    'left_hand_orn': left_hand_orn,
                    'left_hand_pix': left_wrist_xy,
                    'left_hand_pos': left_hand_pos
                }
        r_left = R.from_rotvec(np.array(left_hand_orn))
        print("left hand RPY: ", r_left.as_euler('xyz', degrees=True))
        print("left_wrist_xy: ", left_wrist_xy)
    
    if len(pred_output_list['right_hand'].keys()) > 0:
        right_hand_orn = np.array(pred_output_list['right_hand']['pred_hand_pose'])[0][:3]
        right_wrist_xy = np.array(pred_output_list['right_hand']['pred_joints_img'][0][:2])
        right_wrist_xy = np.array([round(x) for x in right_wrist_xy])

        if right_wrist_xy[1] < depth_img.shape[0] and right_wrist_xy[0] < depth_img.shape[1]:
            depth_val = find_non_zero_depth(depth_img, right_wrist_xy)
            if depth_val is not None:
                print("depth value: ", depth_img[right_wrist_xy[1],right_wrist_xy[0]], depth_val)
                print("right_wrist_xy: ", right_wrist_xy)
                right_hand_pos = get_XYZ(depth_val, right_wrist_xy[0], right_wrist_xy[1])

                hand_info_dict["right"][image_path] = {
                    'color_img': img_original_bgr,
                    'depth_img': depth_img,
                    'right_hand_orn': right_hand_orn,
                    'right_hand_pix': right_wrist_xy,
                    'right_hand_pos': right_hand_pos
                }
        r_right = R.from_rotvec(np.array(right_hand_orn))
        print("right hand RPY: ", r_right.as_euler('xyz', degrees=True))
        print("right_wrist_xy: ", right_wrist_xy)
        
    return hand_info_dict

def main():
    args = config_parser().parse_args()
    depth_folder_path = f"data/{args.folder_name}/depth"
    rgb_folder_path = f"data/{args.folder_name}/color_img"
    path_to_frankmocap = "/home/arpit/test_projects/frankmocap"
    frankmocap_output_path = f"{path_to_frankmocap}/mocap_output/{args.folder_name}/mocap"
    filenames = sorted([f for f in os.listdir(depth_folder_path) if f.endswith('.pickle')])
    hand_info_dict = {"left": {}, "right": {}}
    
    for filename in filenames:
        print("filename: ", filename)
        filename_no_ext = filename.split('.')[0]
        if os.path.isfile(f"{frankmocap_output_path}/{filename_no_ext}_prediction_result.pkl"):
            with open(f"{frankmocap_output_path}/{filename_no_ext}_prediction_result.pkl", 'rb') as handle:
                frankmocap_output = pickle.load(handle)
            with open(f"{depth_folder_path}/{filename}", 'rb') as handle:
                depth_img = pickle.load(handle)
            rgb_file_path = f"{rgb_folder_path}/{filename}"
            img_original_bgr  = cv2.imread(rgb_file_path)
            hand_info_dict = get_hand_info(img_original_bgr,
                                             depth_img,
                                             pred_output_list=frankmocap_output['pred_output_list'][0],
                                             image_path=rgb_file_path,
                                             hand_info_dict=hand_info_dict)

    with open(f'data/{args.folder_name}/hand_poses.pickle', 'wb') as handle:
        pickle.dump(hand_info_dict, handle)



if __name__ == "__main__":
    main()