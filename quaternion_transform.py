import torch
import torchgeometry as tgm
from lib.models.smpl import SMPL_MODEL_DIR#, SMPL
from smplx import SMPL # Original SMPL model
import os
import os.path as osp
import joblib
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()

    path = osp.join('./output', os.path.basename(args.filename).replace('.mp4', ''))
    data = joblib.load(f'{path}/mpsnet_output.pkl')[1]
    print(data.keys())  # Display contents

    # Get SMPL params
    smpl_model = SMPL(
        SMPL_MODEL_DIR,
        batch_size=1,
        create_transl=False,
        gender='neutral'
    )

    # betas = torch.tensor(data["betas"], dtype=torch.float32)
    axis_angle = torch.tensor(data["joints3d"], dtype=torch.float32)

    # Transform joints to quaternions
    print("Axis-Angle Tensor Shape:", axis_angle.shape)
    quats = tgm.angle_axis_to_quaternion(axis_angle)
    print("Quaternion Tensor Shape:", quats.shape)
    print(f"Saving quaternion results to \'{os.path.join(path, 'mpsnet_quats.pkl')}\'.")
    joblib.dump(quats, os.path.join(path, "mpsnet_output.pkl"))
