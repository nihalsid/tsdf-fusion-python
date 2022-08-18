import pickle
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from util.misc import erode_mask


def read_data_from_disk(masks, depths, annots, max_depth, voxel_size):
    print("Estimating voxel volume bounds...")
    n_imgs = len(masks)
    vol_bnds = np.zeros((3, 2))

    depth_im_0 = np.load(str(depths[0]), -1)['arr'].astype(float)
    uv = np.stack(np.meshgrid(np.arange(depth_im_0.shape[1]), np.arange(depth_im_0.shape[0])), -1)
    uv = uv.reshape(-1, 2)
    uvh = np.concatenate([uv, np.ones((len(uv), 1))], -1)
    all_world_points = []
    for i in tqdm(range(n_imgs)):
        mask = erode_mask(cv2.imread(str(masks[i]), -1)).astype(bool)
        depth_im = np.load(str(depths[i]), -1)['arr'].astype(float)
        depth_im = np.nan_to_num(depth_im, nan=0.0, posinf=0.0, neginf=0.0)
        depth_im[~mask] = 0
        depth_im[depth_im > max_depth] = 0
        annotation = pickle.load(open(str(annots[i]), 'rb'))
        intrinsic = np.array(annotation['cam_K'], dtype=np.float32)
        cam2world = np.array(annotation['cam2world_matrix'], dtype=np.float32) @ np.diag([1, -1, -1, 1])
        # Compute camera view frustum and extend convex hull
        cam_point = (np.linalg.inv(intrinsic) @ uvh.T).T * depth_im.reshape(-1, 1)
        world_point = (cam2world[:3, :3] @ cam_point.T).T + cam2world[:3, 3]
        world_point = world_point[depth_im.reshape(-1) != 0, :].T
        if world_point.shape[1] != 0:
            vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(world_point, axis=1))
            vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(world_point, axis=1))
        all_world_points.append(world_point.T)
    vol_bnds[:, 0] -= 5 * voxel_size
    vol_bnds[:, 1] += 5 * voxel_size
    return all_world_points, vol_bnds


def to_voxel_coords(points, bounds, voxel_size, shape):
    world_point = points - bounds
    world_point_vox = torch.from_numpy(world_point / voxel_size).round().long()
    world_point_vox[:, 0], world_point_vox[:, 1], world_point_vox[:, 2] = world_point_vox[:, 0].clip(0, shape[0]), world_point_vox[:, 1].clip(0, shape[1]), world_point_vox[:, 2].clip(0, shape[2])
    return world_point_vox


def check_occupancy(root_folder, max_coverage, num_sets):
    # get all frames
    voxel_size = 0.01
    device = torch.device("cuda:0")
    mask_dir = root_folder / "room_mask"
    depth_dir = root_folder / "depth_npz"
    annot_dir = root_folder / "annotation"

    masks = sorted([x for x in Path(mask_dir).iterdir()], key=lambda x: x.name)
    annots = [Path(annot_dir) / f"{x.stem}.pkl" for x in masks]
    depths = [Path(depth_dir) / f"{x.stem}.npz" for x in masks]

    all_points, vol_bnds = read_data_from_disk(masks, depths, annots, 5, voxel_size)
    vol_dim = np.ceil((vol_bnds[:, 1] - vol_bnds[:, 0]) / voxel_size).copy(order='C').astype(int)
    voxels = torch.zeros(vol_dim.tolist()).bool().to(device)

    mean_height = np.mean([all_points[i][:, 2].mean() for i in range(len(all_points))])

    all_points = [all_points[i][all_points[i][:, 2] < 0.85 * mean_height * 2, :] for i in range(len(all_points))]

    for i in tqdm(range(len(all_points))):
        world_point_vox = to_voxel_coords(all_points[i], vol_bnds[:, 0][None, :], voxel_size, voxels.shape)
        voxels[world_point_vox[:, 0], world_point_vox[:, 1], world_point_vox[:, 2]] = True

    exported_sets = []
    for k in range(num_sets):
        last_incomplete_voxels = torch.zeros(vol_dim.tolist()).bool().to(device)
        current_incomplete_voxels = torch.zeros(vol_dim.tolist()).bool().to(device)
        frame_indices = list(range(len(all_points)))
        random.shuffle(frame_indices)
        used_frames = []
        for i in tqdm(frame_indices):
            last_incomplete_voxels[:] = current_incomplete_voxels[:]
            world_point_vox = to_voxel_coords(all_points[i], vol_bnds[:, 0][None, :], voxel_size, voxels.shape)
            last_incomplete_voxels[world_point_vox[:, 0], world_point_vox[:, 1], world_point_vox[:, 2]] = True
            if last_incomplete_voxels.sum() / voxels.sum() <= max_coverage:
                used_frames.append(i)
                current_incomplete_voxels[:] = last_incomplete_voxels[:]
        vvox = torch.where(current_incomplete_voxels)
        # Path(f"max_cov_{max_coverage:.2f}_{k:02d}.obj").write_text(
        #     "\n".join([f'v {vvox[0][i]} {vvox[1][i]} {vvox[2][i]}' for i in range(vvox[0].shape[0])])
        # )
        print('Frames:', len(used_frames), '/', len(frame_indices))
        exported_sets.append([masks[i].stem for i in used_frames])
    (root_folder / "incomplete").mkdir(exist_ok=True)
    pickle.dump(exported_sets, open(root_folder / "incomplete" / f"cov_{max_coverage:.2f}.pkl", "wb"))
    return exported_sets


if __name__ == "__main__":
    _root = Path("/home/yawarnihal/workspace/nerf-lightning/data/vfront/01ba1742-4fa5-4d1e-8ba4-2f807fe6b283_LivingDiningRoom-4271/")
    _frame_sets = check_occupancy(_root, 0.75, 5)

