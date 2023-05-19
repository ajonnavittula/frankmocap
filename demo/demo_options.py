# Copyright (c) Facebook, Inc. and its affiliates.

import argparse

class DemoOptions():

    def __init__(self):
        parser = argparse.ArgumentParser()
        
        # parser.add_argument('--checkpoint', required=False, default=default_checkpoint, help='Path to pretrained checkpoint')
        default_checkpoint_body_smpl ='./frankmocap/extra_data/body_module/pretrained_weights/2020_05_31-00_50_43-best-51.749683916568756.pt'
        parser.add_argument('--checkpoint_body_smpl', required=False, default=default_checkpoint_body_smpl, help='Path to pretrained checkpoint')
        default_checkpoint_body_smplx ='./frankmocap/extra_data/body_module/pretrained_weights/smplx-03-28-46060-w_spin_mlc3d_46582-2089_2020_03_28-21_56_16.pt'
        parser.add_argument('--checkpoint_body_smplx', required=False, default=default_checkpoint_body_smplx, help='Path to pretrained checkpoint')
        default_checkpoint_hand = "./frankmocap/extra_data/hand_module/pretrained_weights/pose_shape_best.pth"
        parser.add_argument('--checkpoint_hand', required=False, default=default_checkpoint_hand, help='Path to pretrained checkpoint')

        # input options
        parser.add_argument('--input_path', type=str, default=None, help="""Path of video, image, or a folder where image files exists""")
        parser.add_argument('--start_frame', type=int, default=0, help='given a sequence of frames, set the starting frame')
        parser.add_argument('--end_frame', type=int, default=float('inf'), help='given a sequence of frames, set the last frame')
        parser.add_argument('--pkl_dir', type=str, help='Path of storing pkl files that store the predicted results')
        parser.add_argument('--openpose_dir', type=str, help='Directory of storing the prediction of openpose prediction')

        # output options
        parser.add_argument('--out_dir', type=str, default=None, help='Folder of output images.')
        # parser.add_argument('--pklout', action='store_true', help='Export mocap output as pkl file')
        parser.add_argument('--save_bbox_output', action='store_true', help='Save the bboxes in json files (bbox_xywh format)')
        parser.add_argument('--save_pred_pkl', action='store_true', help='Save the predictions (bboxes, params, meshes in pkl format')
        parser.add_argument("--save_mesh", action='store_true', help="Save the predicted vertices and faces")
        parser.add_argument("--save_frame", action='store_true', help='Save the extracted frames from video input or webcam')

        # Other options
        parser.add_argument('--single_person', action='store_true', help='Reconstruct only one person in the scene with the biggest bbox')
        parser.add_argument('--no_display', action='store_true', help='Do not visualize output on the screen')
        parser.add_argument('--no_video_out', action='store_true', help='Do not merge rendered frames to video (ffmpeg)')
        parser.add_argument('--smpl_dir', type=str, default='./frankmocap/extra_data/smpl/', help='Folder where smpl files are located.')
        parser.add_argument('--skip', action='store_true', help='Skip there exist already processed outputs')
        parser.add_argument('--video_url', type=str, default=None, help='URL of YouTube video, or image.')
        parser.add_argument('--download', '-d', action='store_true', help='Download YouTube video first (in webvideo folder), and process it')

        # Body mocap specific options
        parser.add_argument('--use_smplx', action='store_true', help='Use SMPLX model for body mocap')

        # Hand mocap specific options
        parser.add_argument('--view_type', type=str, default='third_view', choices=['third_view', 'ego_centric'],
            help = "The view type of input. It could be ego-centric (such as epic kitchen) or third view")
        parser.add_argument('--crop_type', type=str, default='no_crop', choices=['hand_crop', 'no_crop'],
            help = """ 'hand_crop' means the hand are central cropped in input. (left hand should be flipped to right). 
                        'no_crop' means hand detection is required to obtain hand bbox""")
        
        # Whole motion capture (FrankMocap) specific options
        parser.add_argument('--frankmocap_fast_mode', action='store_true', help="Use fast hand detection mode for whole body motion capture (frankmocap)")

        # renderer
        parser.add_argument("--renderer_type", type=str, default="pytorch3d", 
            choices=['pytorch3d', 'opendr', 'opengl_gui', 'opengl'], help="type of renderer to use")

        # arguments to run with hilvil
        parser.add_argument("--hand", type=str, default="right", 
            choices=['right_hand', 'left_hand'], help="hand to track")
        parser.add_argument("--data-dir", type=str, default="./data", 
            help="Path for parent folder with images and depth info")


        # arguments to run hand object detector
        parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
        parser.add_argument('--cfg', dest='cfg_file',
                            help='optional config file',
                            default='./hand_object_detector/cfgs/res101.yml', type=str)
        parser.add_argument('--net', dest='net',
                            help='vgg16, res50, res101, res152',
                            default='res101', type=str)
        parser.add_argument('--set', dest='set_cfgs',
                            help='set config keys', default=None,
                            nargs=argparse.REMAINDER)
        parser.add_argument('--load_dir', dest='load_dir',
                            help='directory to load models',
                            default="./hand_object_detector/models")
        parser.add_argument('--cuda', dest='cuda', 
                            help='whether use CUDA',
                            action='store_true')
        parser.add_argument('--cag', dest='class_agnostic',
                            help='whether perform class_agnostic bbox regression',
                            action='store_true')
        parser.add_argument('--parallel_type', dest='parallel_type',
                            help='which part of model to parallel, 0: all, 1: model before roi pooling',
                            default=0, type=int)
        parser.add_argument('--checksession', dest='checksession',
                            help='checksession to load model',
                            default=1, type=int)
        parser.add_argument('--checkepoch', dest='checkepoch',
                            help='checkepoch to load network',
                            default=8, type=int)
        parser.add_argument('--checkpoint', dest='checkpoint',
                            help='checkpoint to load network',
                            default=132028, type=int)
        parser.add_argument('--bs', dest='batch_size',
                            help='batch_size',
                            default=1, type=int)
        parser.add_argument('--vis', dest='vis',
                            help='visualization mode',
                            default=True)
        parser.add_argument('--thresh_hand',
                            type=float, default=0.5,
                            required=False)
        parser.add_argument('--thresh_obj', default=0.5,
                            type=float,
                            required=False)

        # params to run object tags
        parser.add_argument("--config", type=str, default="./GroundedSegmentAnything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                            help="path to config file")
        parser.add_argument("--grounded_checkpoint", type=str, default="./GroundedSegmentAnything/groundingdino_swint_ogc.pth",
                            help="path to checkpoint file")
        parser.add_argument("--tag2text_checkpoint", type=str, default="./GroundedSegmentAnything/tag2text_swin_14m.pth",
                            help="path to checkpoint file")
        parser.add_argument("--sam_checkpoint", type=str, default="./GroundedSegmentAnything/sam_vit_b_01ec64.pth", help="path to checkpoint file")
        parser.add_argument("--split", default=",", type=str, help="split for text prompt")
        parser.add_argument("--box_threshold", type=float, default=0.25, help="box threshold")
        parser.add_argument("--text_threshold", type=float, default=0.2, help="text threshold")
        parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")

        parser.add_argument("--device", type=str, default="cuda", help="running on cpu only!, default=False")

        self.parser = parser
    

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt
