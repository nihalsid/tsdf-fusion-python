"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time
from pathlib import Path
import pickle
import cv2
import numpy as np

from fusion.instance_fusion import fuse_inconsistent_instances
from util.front_to_nyu_subset import read_front_semantics
from util.misc import erode_mask, create_box, meshwrite, pcwrite


def get_volume_bounds(masks, depths, annots, max_depth, voxel_size):
    print("Estimating voxel volume bounds...")
    n_imgs = len(masks)
    vol_bnds = np.zeros((3, 2))

    depth_im_0 = np.load(str(depths[0]), -1)['arr'].astype(float)
    uv = np.stack(np.meshgrid(np.arange(depth_im_0.shape[1]), np.arange(depth_im_0.shape[0])), -1)
    uv = uv.reshape(-1, 2)
    uvh = np.concatenate([uv, np.ones((len(uv), 1))], -1)

    for i in range(n_imgs):
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
    vol_bnds[:, 0] -= 5 * voxel_size
    vol_bnds[:, 1] += 5 * voxel_size
    return vol_bnds


def fuse_room_colors(rootdir, vol_bnds, images, masks, depths, annots, vis_dirname, voxel_size):
    from fusion import color_fusion
    n_imgs = len(images)
    print("Initializing voxel volume...")
    tsdf_vol = color_fusion.TSDFVolume(vol_bnds, voxel_size, use_gpu=True)

    # Loop through RGB-D images and fuse them together
    t0_elapse = time.time()
    for i in range(n_imgs):
        print("Fusing frame %d/%d" % (i + 1, n_imgs))
        # Read RGB-D image and camera pose
        mask = cv2.imread(str(masks[i]), -1).astype(bool)
        color_image = cv2.cvtColor(cv2.imread(str(images[i])), cv2.COLOR_BGR2RGB)
        color_image[~mask, :] = 0
        depth_im = np.load(str(depths[i]), -1)['arr'].astype(float)
        depth_im = np.nan_to_num(depth_im, nan=0.0, posinf=0.0, neginf=0.0)
        depth_im[~mask] = 0
        annotation = pickle.load(open(str(annots[i]), 'rb'))
        intrinsic = np.array(annotation['cam_K'], dtype=np.float32)
        cam2world = np.array(annotation['cam2world_matrix'], dtype=np.float32) @ np.diag([1, -1, -1, 1])
        # Integrate observation into voxel volume (assume color aligned with depth)
        tsdf_vol.integrate(color_image, depth_im, intrinsic, cam2world, obs_weight=1.)

    fps = n_imgs / (time.time() - t0_elapse)
    print("Average FPS: {:.2f}".format(fps))

    # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving mesh...")
    (rootdir / "sdf" / vis_dirname).mkdir(exist_ok=True, parents=True)
    verts, verts_vox, faces, norms, colors = tsdf_vol.get_mesh()
    meshwrite(rootdir / "sdf" / vis_dirname / "rgb.ply", verts, faces, norms, colors)
    meshwrite(rootdir / "sdf" / vis_dirname / "rgb_vox_space.ply", verts_vox, faces, norms, colors)

    # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving point cloud to pc.ply...")
    point_cloud = tsdf_vol.get_point_cloud()
    pcwrite(rootdir / "sdf" / vis_dirname / "rgb_pc.ply", point_cloud)

    sdf_vol, col_vol = tsdf_vol.get_volume()
    print('SDF stats:', sdf_vol.min(), sdf_vol.max(), sdf_vol.mean(), sdf_vol.std())
    sdf_vol = sdf_vol[None, ...]
    colors_b = np.floor(col_vol / tsdf_vol._color_const)
    colors_g = np.floor((col_vol - colors_b * tsdf_vol._color_const) / 256)
    colors_r = col_vol - colors_b * tsdf_vol._color_const - colors_g * 256
    col_vol = np.floor(np.asarray([colors_r, colors_g, colors_b]))
    print(sdf_vol.shape, col_vol.shape)

    return sdf_vol, col_vol, tsdf_vol._vol_origin, vol_bnds[:, 1]


def fuse_room_semantics(rootdir, vol_bnds, sems, masks, depths, annots, vis_dirname, voxel_size):
    from fusion import semantic_fusion
    print("Estimating voxel volume bounds...")
    n_imgs = len(masks)

    print("Initializing voxel volume...")
    tsdf_vol = semantic_fusion.TSDFVolume(vol_bnds, voxel_size, n_classes=41, use_gpu=False)

    # Loop through RGB-D images and fuse them together
    t0_elapse = time.time()
    for i in range(n_imgs):
        print("Fusing frame %d/%d" % (i + 1, n_imgs))
        # Read RGB-D image and camera pose
        mask = cv2.imread(str(masks[i]), -1).astype(bool)
        semantics = read_front_semantics(sems[i], mask)
        depth_im = np.load(str(depths[i]), -1)['arr'].astype(float)
        depth_im = np.nan_to_num(depth_im, nan=0.0, posinf=0.0, neginf=0.0)
        depth_im[~mask] = 0
        annotation = pickle.load(open(str(annots[i]), 'rb'))
        intrinsic = np.array(annotation['cam_K'], dtype=np.float32)
        cam2world = np.array(annotation['cam2world_matrix'], dtype=np.float32) @ np.diag([1, -1, -1, 1])
        # Integrate observation into voxel volume (assume color aligned with depth)
        tsdf_vol.integrate(semantics, depth_im, intrinsic, cam2world, obs_weight=1.)

    fps = n_imgs / (time.time() - t0_elapse)
    print("Average FPS: {:.2f}".format(fps))

    (rootdir / "sdf" / vis_dirname).mkdir(exist_ok=True, parents=True)

    # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving mesh to mesh.ply...")
    verts, verts_vox, faces, norms, colors = tsdf_vol.get_mesh()

    meshwrite(rootdir / "sdf" / vis_dirname / "sem.ply", verts, faces, norms, colors)
    meshwrite(rootdir / "sdf" / vis_dirname / "sem_vox_space.ply", verts_vox, faces, norms, colors)

    # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving point cloud to pc.ply...")
    point_cloud = tsdf_vol.get_point_cloud()
    pcwrite(rootdir / "sdf" / vis_dirname / "sem_pc.ply", point_cloud)

    sdf_vol, col_vol = tsdf_vol.get_volume()
    print('SDF stats:', sdf_vol.min(), sdf_vol.max(), sdf_vol.mean(), sdf_vol.std())
    sdf_vol = sdf_vol[None, ...]
    col_vol = col_vol.transpose((3, 0, 1, 2))
    print(sdf_vol.shape, col_vol.shape)
    return sdf_vol, col_vol, tsdf_vol._vol_origin, vol_bnds[:, 1]


def run_fuse_colors_and_semantics(room_root, resolution, max_depth):
    images = sorted([x for x in Path(room_root / "rgb").iterdir()], key=lambda x: x.name)
    masks = [Path(room_root / "room_mask") / x.name for x in images]
    annots = [Path(room_root / "annotation") / f"{x.stem}.pkl" for x in images]
    depths = [Path(room_root / "depth_npz") / f"{x.stem}.npz" for x in images]
    sems = [Path(room_root / "sem") / f"{x.stem}.png" for x in images]

    vol_bnds = get_volume_bounds(masks, depths, annots, max_depth, resolution)

    vis_dirname = "visualization"
    sdf_vol, col_vol, vol_origin, vol_end = fuse_room_colors(room_root, np.copy(vol_bnds), images, masks, depths, annots, vis_dirname, resolution)
    # _, sem_vol, _, _ = fuse_room_semantics(room_root, np.copy(vol_bnds), sems, masks, depths, annots, vis_dirname, resolution)
    create_box((vol_bnds[:, 1] + vol_bnds[:, 0]) / 2, vol_bnds[:, 1] - vol_bnds[:, 0]).export(room_root / "sdf" / vis_dirname / "bounds.obj")
    # np.savez_compressed(room_root / "sdf" / f"{resolution:.2f}.npz", sdf=sdf_vol, color=col_vol, semantics=sem_vol.argmax(0), volume_origin=vol_origin, voxel_size=resolution, volume_end=vol_end)

    max_coverage = 0.75
    frame_sets = pickle.load(open(room_root / "incomplete" / f"cov_{max_coverage:.2f}.pkl", "rb"))
    for idx, frame_set in enumerate(frame_sets):
        vis_dirname = f"visualization_{max_coverage:.2f}_{idx:02}"
        images_set = [x for x in images if x.stem in frame_set]
        masks_set = [x for x in masks if x.stem in frame_set]
        depths_set = [x for x in depths if x.stem in frame_set]
        annots_set = [x for x in annots if x.stem in frame_set]
        fuse_room_colors(room_root, np.copy(vol_bnds), images_set, masks_set, depths_set, annots_set, vis_dirname, resolution)


def run_fuse_instances(room_root, resolution):
    inst_vol = fuse_inconsistent_instances(room_root, resolution)
    npz = np.load(room_root / "sdf" / f"{resolution:.2f}.npz")
    sdf_vol, col_vol, sem_vol, vol_origin, vol_end = npz['sdf'], npz['color'], npz['semantics'], npz['volume_origin'], npz['volume_end']
    np.savez_compressed(room_root / "sdf" / f"{resolution:.2f}.npz", sdf=sdf_vol, color=col_vol, semantics=sem_vol.argmax(0), instances=inst_vol, volume_origin=vol_origin, voxel_size=resolution, volume_end=vol_end)


if __name__ == "__main__":
    _resolution = 0.05
    _max_depth = 5
    _root = Path("/home/yawarnihal/workspace/nerf-lightning/data/vfront/01ba1742-4fa5-4d1e-8ba4-2f807fe6b283_LivingDiningRoom-4271/")
    run_fuse_colors_and_semantics(_root, _resolution, _max_depth)
    # run_fuse_instances(_root, _resolution)
