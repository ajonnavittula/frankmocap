# Copyright (c) Facebook, Inc. and its affiliates.

import os, sys, shutil
import os.path as osp
import numpy as np
import cv2
import json
import torch
from torchvision.transforms import Normalize

from demo.demo_options import DemoOptions
import mocap_utils.general_utils as gnu
import mocap_utils.demo_utils as demo_utils

from handmocap.hand_mocap_api import HandMocap
from handmocap.hand_bbox_detector import HandBboxDetector

import renderer.image_utils as imu
from renderer.viewer2D import ImShow
import time, pickle
import re
from tqdm import tqdm


def num_sort(input_string):
    return list(map(int, re.findall(r'\d+', input_string)))[0]

def get_hand_traj(args):
    # demo_args = DemoOptions().parse()
    args.use_smplx = True

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    assert torch.cuda.is_available(), "Current version only supports GPU"

    #Set Bbox detector
    bbox_detector =  HandBboxDetector(args.view_type, device)

    # Set Mocap regressor
    hand_mocap = HandMocap(args.checkpoint_hand, args.smpl_dir, device = device)
    # Set Visualizer
    assert args.renderer_type == "pytorch3d", "Only supports pytorch3d as visualizer"
    if args.renderer_type in ['pytorch3d', 'opendr']:
        from renderer.screen_free_visualizer import Visualizer
    else:
        from renderer.visualizer import Visualizer
    visualizer = Visualizer(args.renderer_type)

    # start regress
    image_dir = os.path.join(os.path.abspath(args.data_dir), "rgb")
    input_data = sorted(os.listdir(image_dir), key=num_sort)
    assert len(input_data) > 0, "No images found in the directory"
    print("Found {} images in directory".format(len(input_data)))
    
    hand_traj = []
    for image_idx in tqdm(range(len(input_data)), dynamic_ncols=True):
        image_num = input_data[image_idx]
        # print("loading img {}".format(image_num))
        image_path = os.path.join(image_dir, image_num)
        img_original_bgr = cv2.imread(image_path)

        detect_output = bbox_detector.detect_hand_bbox(img_original_bgr.copy())
        body_pose_list, body_bbox_list, hand_bbox_list, raw_hand_bboxes = detect_output

        if len(hand_bbox_list) < 1:
            print(f"No hand deteced: {image_path}")
            continue
    
        # Hand Pose Regression
        pred_output_list = hand_mocap.regress(
                img_original_bgr, hand_bbox_list, add_margin=True)

        if args.hand not in pred_output_list[0]:
            print("{} not found in {}".format(args.hand, image_path))
            continue

        pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)

        # visualize
        res_img = visualizer.visualize(
            img_original_bgr, 
            pred_mesh_list = pred_mesh_list, 
            hand_bbox_list = hand_bbox_list)
        
        demo_utils.save_res_img(args.data_dir, image_path, res_img)

        # get pixel coords and depth
        # we get the hand from the list and then the wrist (x,y) position from hand
        pixel_coords = pred_output_list[0][args.hand]["pred_joints_img"][0][:2]
        pixel_coords = [int(coords) for coords in pixel_coords]
        depth_filename = os.path.splitext(image_num)[0] + ".pkl"
        depth_path = os.path.join(args.data_dir, "depth", depth_filename)
        depth_data = pickle.load(open(depth_path, "rb"), encoding="latin1")
        depth = np.mean(depth_data[pixel_coords[0]-1:pixel_coords[0]+1, pixel_coords[1]-1:pixel_coords[1]+1])

        if depth == 0:
            print("Depth information corrupted for {}, ignoring image".format(image_path))
            continue
        orientation = pred_output_list[0][args.hand]["pred_hand_pose"][0, :3]
        orientation = orientation.squeeze()
        point = dict()
        point["pixel_coords"] = pixel_coords
        point["depth"] = depth
        point["orientation"] = orientation
        point["image_path"] = image_path
        point["depth_path"] = depth_path

        hand_traj.append(point)
    
    print("Found a total of {} points in video".format(len(hand_traj)))
    return hand_traj


def run_hand_mocap(args, bbox_detector, hand_mocap, visualizer):
    #Set up input data (images or webcam)
    input_type, input_data = demo_utils.setup_input(args)
 
    assert args.out_dir is not None, "Please specify output dir to store the results"
    cur_frame = args.start_frame
    video_frame = 0

    wrist_pos = []
    while True:
        # load data
        load_bbox = False

        if input_type =='image_dir':
            if cur_frame < len(input_data):
                image_path = input_data[cur_frame]
                img_original_bgr  = cv2.imread(image_path)
                print("Loaded image: {}".format(image_path))
            else:
                img_original_bgr = None

        elif input_type == 'bbox_dir':
            if cur_frame < len(input_data):
                print("Use pre-computed bounding boxes")
                image_path = input_data[cur_frame]['image_path']
                hand_bbox_list = input_data[cur_frame]['hand_bbox_list']
                body_bbox_list = input_data[cur_frame]['body_bbox_list']
                img_original_bgr  = cv2.imread(image_path)
                load_bbox = True
            else:
                img_original_bgr = None

        elif input_type == 'video':      
            _, img_original_bgr = input_data.read()
            if video_frame < cur_frame:
                video_frame += 1
                continue
            # save the obtained video frames
            image_path = osp.join(args.out_dir, "frames", f"{cur_frame:05d}.jpg")
            if img_original_bgr is not None:
                video_frame += 1
                if args.save_frame:
                    gnu.make_subdir(image_path)
                    cv2.imwrite(image_path, img_original_bgr)
        
        elif input_type == 'webcam':
            _, img_original_bgr = input_data.read()

            if video_frame < cur_frame:
                video_frame += 1
                continue
            # save the obtained video frames
            image_path = osp.join(args.out_dir, "frames", f"scene_{cur_frame:05d}.jpg")
            if img_original_bgr is not None:
                video_frame += 1
                if args.save_frame:
                    gnu.make_subdir(image_path)
                    cv2.imwrite(image_path, img_original_bgr)
        else:
            assert False, "Unknown input_type"

        cur_frame +=1
        if img_original_bgr is None or cur_frame > args.end_frame:
            break   
        print("--------------------------------------")

        # bbox detection
        if load_bbox:
            body_pose_list = None
            raw_hand_bboxes = None
        elif args.crop_type == 'hand_crop':
            # hand already cropped, thererore, no need for detection
            img_h, img_w = img_original_bgr.shape[:2]
            body_pose_list = None
            raw_hand_bboxes = None
            hand_bbox_list = [ dict(right_hand = np.array([0, 0, img_w, img_h])) ]
        else:            
            # Input images has other body part or hand not cropped.
            # Use hand detection model & body detector for hand detection
            assert args.crop_type == 'no_crop'
            detect_output = bbox_detector.detect_hand_bbox(img_original_bgr.copy())
            body_pose_list, body_bbox_list, hand_bbox_list, raw_hand_bboxes = detect_output
            print("BBox detected")
            # print(body_pose_list)
        
        # save the obtained body & hand bbox to json file
        if args.save_bbox_output:
            demo_utils.save_info_to_json(args, image_path, body_bbox_list, hand_bbox_list)

        if len(hand_bbox_list) < 1:
            print(f"No hand deteced: {image_path}")
            continue
    
        # Hand Pose Regression
        pred_output_list = hand_mocap.regress(
                img_original_bgr, hand_bbox_list, add_margin=True)
        print("Hand regressed")
        try:
            # print(len(pred_output_list))
            # if pred_output_list[0]["right_hand"]["pred_joints_img"][0]:
            wrist_pos.append([pred_output_list, image_path])
            # print(pred_output_list[0]["right_hand"]["pred_joints_smpl"][0])
            # wrist_pos.append(pred_output_list[0]["right_hand"]["pred_joints_smpl"][0].tolist())
        except:
            pass
        # print(pred_output_list[0]["left_hand"]["pred_hand_pose"])
        # print(pred_output_list[0]["left_hand"]["pred_joints_img"][0])
        assert len(hand_bbox_list) == len(body_bbox_list)
        assert len(body_bbox_list) == len(pred_output_list)

        # extract mesh for rendering (vertices in image space and faces) from pred_output_list
        pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)
        print("Mesh extracted")

        # visualize
        res_img = visualizer.visualize(
            img_original_bgr, 
            pred_mesh_list = pred_mesh_list, 
            hand_bbox_list = hand_bbox_list)
        print("Visualized")
        # show result in the screen
        if not args.no_display:
            res_img = res_img.astype(np.uint8)
            ImShow(res_img)

        # save the image (we can make an option here)
        if args.out_dir is not None:
            demo_utils.save_res_img(args.out_dir, image_path, res_img)

        # save predictions to pkl
        if args.save_pred_pkl:
            demo_type = 'hand'
            demo_utils.save_pred_to_pkl(
                args, demo_type, image_path, body_bbox_list, hand_bbox_list, pred_output_list)

        print(f"Processed : {image_path}")
    
    pickle.dump(wrist_pos, open("test.pkl", "wb"))
    #save images as a video
    if not args.no_video_out and input_type in ['video', 'webcam']:
        demo_utils.gen_video_out(args.out_dir, args.seq_name)

    # When everything done, release the capture
    if input_type =='webcam' and input_data is not None:
        input_data.release()
    cv2.destroyAllWindows()

  
def main():
    args = DemoOptions().parse()
    args.use_smplx = True

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    assert torch.cuda.is_available(), "Current version only supports GPU"

    #Set Bbox detector
    bbox_detector =  HandBboxDetector(args.view_type, device)

    # Set Mocap regressor
    hand_mocap = HandMocap(args.checkpoint_hand, args.smpl_dir, device = device)
    # Set Visualizer
    if args.renderer_type in ['pytorch3d', 'opendr']:
        from renderer.screen_free_visualizer import Visualizer
    else:
        from renderer.visualizer import Visualizer
    visualizer = Visualizer(args.renderer_type)

    # run
    run_hand_mocap(args, bbox_detector, hand_mocap, visualizer)
   

if __name__ == '__main__':
    main()
