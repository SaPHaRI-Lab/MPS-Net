import joblib
import argparse
import os
import os.path as osp

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import PathCollection as col
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

connections = [
    # Torso
    (12, 0), (12, 17), (12, 16),  # Neck to MidHip, RShoulder, LShoulder
    (0, 2), (0, 1),  # MidHip to RHip, LHip
    
    # Right Arm
    # (17, 19), (19, 21),  # RShoulder to RElbow to RWrist
    
    # # Left Arm
    # (16, 18), (18, 20),  # LShoulder to LElbow to LWrist
    
    # # Right Leg
    # (2, 5), (5, 8),  # RHip to RKnee to RAnkle
    
    # # Left Leg
    # (1, 4), (4, 7),  # LHip to LKnee to LAnkle
    
    # # Head
    # (12, 24),  # Neck to Nose to REye/LEye
    #   # Eyes to Ears
    
    # # Feet
    # (8, 34), (34, 32), (34, 33),  # RAnkle to RHeel, RBigToe to RSmallToe
    # (7, 31), (31, 29), (31, 30),  # LAnkle to LHeel, LBigToe to LSmallToe
]

def plot_3D_joints(joints: np.array, path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # x, y, z = joints[0, :, 0], joints[0, :, 1], joints[0, :, 2]
    # plot = ax.scatter([],[],[], c="r", marker="o")  # Plot joints
    # plot._offsets3d = (joints[0, :, 0], joints[0, :, 1], joints[0, :, 2])
    
    # ax.set_xlim(np.min(joints[:, :, 0]), np.max(joints[:, :, 0]))
    # ax.set_ylim(np.min(joints[:, :, 1]), np.max(joints[:, :, 1]))
    # ax.set_zlim(np.min(joints[:, :, 2]), np.max(joints[:, :, 2]))

    lines = []
    for (start_idx, end_idx) in connections:
        line, = ax.plot(
            [joints[0, start_idx, 0], joints[0, end_idx, 0]],
            [joints[0, start_idx, 1], joints[0, end_idx, 1]],
            [joints[0, start_idx, 2], joints[0, end_idx, 2]],\
            'b-', lw=2
        )
        lines.append(line)
    
    def animate(i):
        # plot._offset3d = (joints[i, :, 0], joints[i, :, 1], joints[i, :, 2])
        for line, (start_idx, end_idx) in zip(lines, connections):
            line.set_data_3d(
                [joints[i, start_idx, 0], joints[i, end_idx, 0]],
                [joints[i, start_idx, 1], joints[i, end_idx, 1]],
                [joints[i, start_idx, 2], joints[i, end_idx, 2]]
            )
        
        return lines,   

    anim = animation.FuncAnimation(fig, animate, 
                        frames=joints.shape[0],
                        interval=10, 
                        repeat_delay=1000,
                        blit=True
    )
    anim.save(f'{path}/3D_animation.mp4', writer='ffmpeg', fps=30)
    plt.show()
    return anim
    
def scatter_3D_joints(joints: np.array, path: str):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Set fixed axis limits to avoid rescaling
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    
    # Initialize scatter plot
    plot = ax.scatter(joints[0, :, 0], joints[0, :, 1], joints[0, :, 2], c="r", marker="o")
    ax.view_init(elev=-75, azim=-90)

    def animate(i):
        plot._offsets3d = (joints[i, :, 0], joints[i, :, 1], joints[i, :, 2])
        return plot,

    anim = animation.FuncAnimation(fig, animate, 
                                   frames=joints.shape[0], 
                                   interval=10, 
                                   repeat_delay=1000, 
                                   blit=False)
    
    anim.save(f'{path}/3D_animation.mp4', writer='ffmpeg', fps=30)
    plt.show()
    return anim

def scatter_2D_joints(joints: np.array, path):
    fig, ax = plt.subplots()
    
    scatter = ax.scatter(joints[0, :, 0], joints[1, :, 1])
    def animate(i):
        scatter.set_offsets(joints[i, :, 0], joints[i, :, 1])
        return scatter,

    anim = animation.FuncAnimation(fig, animate, 
                        frames=joints.shape[0],
                        interval=100, 
                        repeat_delay=1000,
                        blit=False
    )
    # anim.save(f'{path}/2D_animation.mp4', writer='ffmpeg', fps=30)
    plt.show()
    return anim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()

    path = osp.join('./output', os.path.basename(args.filename).replace('.mp4', ''))
    data = joblib.load(f'{path}/mpsnet_output.pkl')[1]
    print(data.keys())  # Display contents

    # MPS Joint Format
    joints3d, joints2d = data['joints3d'], data['joints2d']
    print("Joints Shape:", joints3d.shape)
    # print("First Frame Joints:", joints3d[0])

    # H36M Joint format
    import torch
    from lib.models.smpl import SMPL_MODEL_DIR#, SMPL
    from smplx import SMPL

    smpl_model = SMPL(
        SMPL_MODEL_DIR,
        batch_size=1,
        create_transl=False,
        gender='neutral'
    )

    betas = torch.tensor(data["betas"], dtype=torch.float32)
    pose = torch.tensor(data["pose"], dtype=torch.float32)
    print("Beta, Pose:", betas.shape, pose.shape)

    output = smpl_model(betas=betas, body_pose=pose[:, 3:], global_orient=pose[:, :3])
    joints3d_positions = output.joints
    joint3d_rotations = pose[:, 3:]

    print("Computed Joints Shape:", joints3d_positions.detach().numpy().shape)
    print("Computed Joints Shape:", joints3d.shape)
    # print("Joint Rotations Shape:", joint3d_rotations.shape)    

    # plot_3D_joints(joints3d, path)
    scatter_3D_joints(joints3d, path)
    # scatter_3D_joints(joints3d_positions.detach().numpy(), path)
    # scatter_2D_joints(joints2d, path)