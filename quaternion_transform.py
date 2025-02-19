import torch
from lib.models.smpl import SMPL_MODEL_DIR#, SMPL
from smplx import SMPL # Original SMPL model

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
    axis_angle = torch.tensor(data["pose"], dtype=torch.float32)

    # Tramsform joints to quaternions
    quats = transforms.axis_angle_to_quaternion(axis_angle)
    print(quats)
    # print(f"Saving quaternion results to \'{os.path.join(output_path, 'mpsnet_output.pkl')}\'.")
    # joblib.dump(running_results, os.path.join(output_path, "mpsnet_output.pkl"))