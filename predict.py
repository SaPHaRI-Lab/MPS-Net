import os
import os.path as osp
from lib.core.config import BASE_DATA_DIR
from lib.models.smpl import SMPL, SMPL_MODEL_DIR
##os.environ['PYOPENGL_PLATFORM'] = 'egl'

import cv2
import time
import torch
import joblib
import shutil
import colorsys
import argparse
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multi_person_tracker import MPT
from torch.utils.data import DataLoader

from lib.models.mpsnet import MPSnet
from lib.utils.renderer import Renderer
from lib.dataset._dataset_demo import CropDataset, FeatureDataset
from lib.utils.demo_utils import (
    download_youtube_clip,
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    video_to_images,
    images_to_video,
)

MIN_NUM_FRAMES = 25
random.seed(1)
torch.manual_seed(1)
np.random.seed(1)

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    """ Prepare input video (images) """
    video_file = args.vid_file
    if video_file.startswith('https://www.youtube.com'):
        print(f"Donwloading YouTube video \'{video_file}\'")
        video_file = download_youtube_clip(video_file, '/tmp')
        if video_file is None:
            exit('Youtube url is not valid!')
        print(f"YouTube Video has been downloaded to {video_file}...")

    if not os.path.isfile(video_file):
        exit(f"Input video \'{video_file}\' does not exist!")

    output_path = osp.join('./output', os.path.basename(video_file).replace('.mp4', ''))
    Path(output_path).mkdir(parents=True, exist_ok=True)
    image_folder, num_frames, img_shape = video_to_images(video_file, return_info=True)

    print(f"Input video number of frames {num_frames}\n")
    orig_height, orig_width = img_shape[:2]

    """ Run tracking """
    total_time = time.time()
    bbox_scale = 1.1    #
    # run multi object tracker
    mot = MPT(
        device=device,
        batch_size=args.tracker_batch_size,
        display=args.display,
        detector_type=args.detector,
        output_format='dict',
        yolo_img_size=args.yolo_img_size,
    )
    tracking_results = mot(image_folder)

    # remove tracklets if num_frames is less than MIN_NUM_FRAMES
    for person_id in list(tracking_results.keys()):
        if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
            del tracking_results[person_id]

    """ Get MPSnet model """
    seq_len = 16
    model = MPSnet(
        seqlen=seq_len,
        n_layers=2,
        hidden_size=1024
    ).to(device)

    # Load pretrained weights
    pretrained_file = args.model
    ckpt = torch.load(pretrained_file)
    print(f"Load pretrained weights from \'{pretrained_file}\'")
    ckpt = ckpt['gen_state_dict']
    model.load_state_dict(ckpt, strict=False)

    # Change mesh gender
    gender = args.gender  # 'neutral', 'male', 'female'
    model.regressor.smpl = SMPL(
        SMPL_MODEL_DIR,
        batch_size=64,
        create_transl=False,
        gender=gender
    ).cuda()
    model.eval()

    # Feature_extractor
    from lib.models.spin import hmr
    hmr = hmr().to(device)
    checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
    hmr.load_state_dict(checkpoint['model'], strict=False)
    hmr.eval()
    """ Run MPSnet on each person """
    print("\nRunning MPSnet on each person tracklet...")
    running_time = time.time()
    running_results = {}
    for person_id in tqdm(list(tracking_results.keys())):
        bboxes = joints2d = None
        bboxes = tracking_results[person_id]['bbox']
        frames = tracking_results[person_id]['frames']

        # Prepare static image features
        dataset = CropDataset(
            image_folder=image_folder,
            frames=frames,
            bboxes=bboxes,
            joints2d=joints2d,
            scale=bbox_scale,
        )

        bboxes = dataset.bboxes
        frames = dataset.frames
        has_keypoints = True if joints2d is not None else False

        crop_dataloader = DataLoader(dataset, batch_size=256, num_workers=0)

        with torch.no_grad():
            feature_list = []
            for i, batch in enumerate(crop_dataloader):
                if has_keypoints:
                    batch, nj2d = batch
                    norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

                batch = batch.to(device)
                feature = hmr.feature_extractor(batch.reshape(-1,3,224,224))
                feature_list.append(feature.cpu())

            del batch

            feature_list = torch.cat(feature_list, dim=0)

        # Encode temporal features and estimate 3D human mesh
        dataset = FeatureDataset(
            image_folder=image_folder,
            frames=frames,
            seq_len=seq_len,
        )
        dataset.feature_list = feature_list
        dataloader = DataLoader(dataset, batch_size=64, num_workers=0)

        with torch.no_grad():
            pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [], [], [], [], [], []

            for i, batch in enumerate(dataloader):
                if has_keypoints:
                    batch, nj2d = batch
                    norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

                batch = batch.to(device)
                output = model(batch)[0][-1]

                pred_cam.append(output['theta'][:, :3])
                pred_verts.append(output['verts'])
                pred_pose.append(output['theta'][:, 3:75])
                pred_betas.append(output['theta'][:, 75:])
                pred_joints3d.append(output['kp_3d'])

            pred_cam = torch.cat(pred_cam, dim=0)
            pred_verts = torch.cat(pred_verts, dim=0)
            pred_pose = torch.cat(pred_pose, dim=0)
            pred_betas = torch.cat(pred_betas, dim=0)
            pred_joints3d = torch.cat(pred_joints3d, dim=0)

            del batch

            # ========= Save results to a pickle file ========= #
        pred_cam = pred_cam.cpu().numpy()
        pred_verts = pred_verts.cpu().numpy()
        pred_pose = pred_pose.cpu().numpy()
        pred_betas = pred_betas.cpu().numpy()
        pred_joints3d = pred_joints3d.cpu().numpy()

        bboxes[:, 2:] = bboxes[:, 2:] * 1.2
        if args.render_plain:
            pred_cam[:,0], pred_cam[:,1:] = 1, 0  # np.array([[1, 0, 0]])
        orig_cam = convert_crop_cam_to_orig_img(
            cam=pred_cam,
            bbox=bboxes,
            img_width=orig_width,
            img_height=orig_height
        )

        output_dict = {
            'pred_cam': pred_cam,
            'orig_cam': orig_cam,
            'verts': pred_verts,
            'pose': pred_pose,
            'betas': pred_betas,
            'joints3d': pred_joints3d,
            'joints2d': joints2d,
            'bboxes': bboxes,
            'frame_ids': frames,
        }

        running_results[person_id] = output_dict

    del model
    end = time.time()
    fps = num_frames / (end - running_time)
    print(f'MPSnet FPS: {fps:.2f}')
    total_time = time.time() - total_time
    print(f'Total time spent: {total_time:.2f} seconds (including model loading time).')
    print(f'Total FPS (including model loading time): {num_frames / total_time:.2f}.')

    if args.save_pkl:
        print(f"Saving output results to \'{os.path.join(output_path, 'mpsnet_output.pkl')}\'.")
        joblib.dump(running_results, os.path.join(output_path, "mpsnet_output.pkl"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--vid_file', type=str, default='sample_video.mp4', help='input video path or youtube link')

    parser.add_argument('--model', type=str, default='./data/base_data/mpsnet_model_best.pth.tar', help='path to pretrained model weight')

    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'],
                        help='object detector to be used for bbox tracking')

    parser.add_argument('--yolo_img_size', type=int, default=416,
                        help='input image size for yolo detector')

    parser.add_argument('--tracker_batch_size', type=int, default=12,
                        help='batch size of object detector used for bbox tracking')

    parser.add_argument('--display', action='store_true',
                        help='visualize the results of each step during demo')

    parser.add_argument('--save_pkl', action='store_true',
                        help='save results to a pkl file')

    parser.add_argument('--save_obj', action='store_true',
                        help='save results as .obj files.')

    parser.add_argument('--gender', type=str, default='neutral',
                        help='set gender of people from (neutral, male, female)')

    parser.add_argument('--wireframe', action='store_true',
                        help='render all meshes as wireframes.')

    parser.add_argument('--sideview', action='store_true',
                        help='render meshes from alternate viewpoint.')

    parser.add_argument('--render_plain', action='store_true',
                        help='render meshes on plain background')

    parser.add_argument('--gpu', type=int, default='1', help='gpu num')

    args = parser.parse_args()
    args.save_pkl = True
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    main(args)