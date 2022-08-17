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


def get_volume_bounds(image_dir, mask_dir, depth_dir, annot_dir, max_depth, voxel_size):
    print("Estimating voxel volume bounds...")
    images = sorted([x for x in Path(image_dir).iterdir()], key=lambda x: x.name)
    masks = [Path(mask_dir) / x.name for x in images]
    annots = [Path(annot_dir) / f"{x.stem}.pkl" for x in images]
    depths = [Path(depth_dir) / f"{x.stem}.npz" for x in images]
    n_imgs = len(images)
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
    (image_dir.parent / "sdf" / "visualization").mkdir(exist_ok=True, parents=True)
    create_box((vol_bnds[:, 1] + vol_bnds[:, 0]) / 2, vol_bnds[:, 1] - vol_bnds[:, 0]).export(image_dir.parent / "sdf" / "visualization" / "bounds.obj")
    return vol_bnds


def fuse_room_colors(vol_bnds, image_dir, mask_dir, depth_dir, annot_dir, voxel_size):
    from fusion import color_fusion
    images = sorted([x for x in Path(image_dir).iterdir()], key=lambda x: x.name)
    masks = [Path(mask_dir) / x.name for x in images]
    annots = [Path(annot_dir) / f"{x.stem}.pkl" for x in images]
    depths = [Path(depth_dir) / f"{x.stem}.npz" for x in images]
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
    (image_dir.parent / "sdf" / "visualization").mkdir(exist_ok=True, parents=True)
    verts, verts_vox, faces, norms, colors = tsdf_vol.get_mesh()
    meshwrite(image_dir.parent / "sdf" / "visualization" / "rgb.ply", verts, faces, norms, colors)
    meshwrite(image_dir.parent / "sdf" / "visualization" / "rgb_vox_space.ply", verts_vox, faces, norms, colors)

    # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving point cloud to pc.ply...")
    point_cloud = tsdf_vol.get_point_cloud()
    pcwrite(image_dir.parent / "sdf" / "visualization" / "rgb_pc.ply", point_cloud)

    sdf_vol, col_vol = tsdf_vol.get_volume()
    print('SDF stats:', sdf_vol.min(), sdf_vol.max(), sdf_vol.mean(), sdf_vol.std())
    sdf_vol = sdf_vol[None, ...]
    colors_b = np.floor(col_vol / tsdf_vol._color_const)
    colors_g = np.floor((col_vol - colors_b * tsdf_vol._color_const) / 256)
    colors_r = col_vol - colors_b * tsdf_vol._color_const - colors_g * 256
    col_vol = np.floor(np.asarray([colors_r, colors_g, colors_b]))
    print(sdf_vol.shape, col_vol.shape)

    return sdf_vol, col_vol, tsdf_vol._vol_origin, vol_bnds[:, 1]


def fuse_room_semantics(vol_bnds, sem_dir, mask_dir, depth_dir, annot_dir, voxel_size):
    from fusion import semantic_fusion
    print("Estimating voxel volume bounds...")
    images = sorted([x for x in Path(sem_dir).iterdir()], key=lambda x: x.name)
    masks = [Path(mask_dir) / x.name for x in images]
    annots = [Path(annot_dir) / f"{x.stem}.pkl" for x in images]
    depths = [Path(depth_dir) / f"{x.stem}.npz" for x in images]
    n_imgs = len(images)

    print("Initializing voxel volume...")
    tsdf_vol = semantic_fusion.TSDFVolume(vol_bnds, voxel_size, n_classes=41, use_gpu=False)

    # Loop through RGB-D images and fuse them together
    t0_elapse = time.time()
    for i in range(n_imgs):
        print("Fusing frame %d/%d" % (i + 1, n_imgs))
        # Read RGB-D image and camera pose
        mask = cv2.imread(str(masks[i]), -1).astype(bool)
        semantics = read_front_semantics(images[i], mask)
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

    (sem_dir.parent / "sdf" / "visualization").mkdir(exist_ok=True, parents=True)

    # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving mesh to mesh.ply...")
    verts, verts_vox, faces, norms, colors = tsdf_vol.get_mesh()

    meshwrite(sem_dir.parent / "sdf" / "visualization" / "sem.ply", verts, faces, norms, colors)
    meshwrite(sem_dir.parent / "sdf" / "visualization" / "sem_vox_space.ply", verts_vox, faces, norms, colors)

    # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving point cloud to pc.ply...")
    point_cloud = tsdf_vol.get_point_cloud()
    pcwrite(sem_dir.parent / "sdf" / "visualization" / "sem_pc.ply", point_cloud)

    sdf_vol, col_vol = tsdf_vol.get_volume()
    print('SDF stats:', sdf_vol.min(), sdf_vol.max(), sdf_vol.mean(), sdf_vol.std())
    sdf_vol = sdf_vol[None, ...]
    col_vol = col_vol.transpose((3, 0, 1, 2))
    print(sdf_vol.shape, col_vol.shape)
    return sdf_vol, col_vol, tsdf_vol._vol_origin, vol_bnds[:, 1]


def run_fuse_room_colors_vfront(room_root, resolution, max_depth):
    vol_bnds = get_volume_bounds(room_root / "rgb", room_root / "room_mask", room_root / "depth_npz", room_root / "annotation", max_depth, resolution)
    sdf_vol, col_vol, vol_origin, vol_end = fuse_room_colors(vol_bnds, room_root / "rgb", room_root / "room_mask", room_root / "depth_npz", room_root / "annotation", resolution)
    Path(room_root / "sdf").mkdir(exist_ok=True)
    np.savez_compressed(room_root / "sdf" / f"{resolution:.2f}.npz", sdf=sdf_vol, color=col_vol, volume_origin=vol_origin, voxel_size=resolution, volume_end=vol_end)


def run_fuse_room_semantics_vfront(room_root, resolution, max_depth):
    vol_bnds = get_volume_bounds(room_root / "rgb", room_root / "room_mask", room_root / "depth_npz", room_root / "annotation", max_depth, resolution)
    sdf_vol, sem_vol, vol_origin, vol_end = fuse_room_semantics(vol_bnds, room_root / "sem", room_root / "room_mask", room_root / "depth_npz", room_root / "annotation", resolution)
    col_vol = np.load(room_root / "sdf" / f"{resolution:.2f}.npz")['color']
    np.savez_compressed(room_root / "sdf" / f"{resolution:.2f}.npz", sdf=sdf_vol, color=col_vol, semantics=sem_vol.argmax(0), volume_origin=vol_origin, voxel_size=resolution, volume_end=vol_end)


def run_fuse_colors_and_semantics(room_root, resolution, max_depth):
    vol_bnds = get_volume_bounds(room_root / "rgb", room_root / "room_mask", room_root / "depth_npz", room_root / "annotation", max_depth, resolution)
    sdf_vol, col_vol, vol_origin, vol_end = fuse_room_colors(np.copy(vol_bnds), room_root / "rgb", room_root / "room_mask", room_root / "depth_npz", room_root / "annotation", resolution)
    _, sem_vol, _, _ = fuse_room_semantics(np.copy(vol_bnds), room_root / "sem", room_root / "room_mask", room_root / "depth_npz", room_root / "annotation", resolution)
    np.savez_compressed(room_root / "sdf" / f"{resolution:.2f}.npz", sdf=sdf_vol, color=col_vol, semantics=sem_vol.argmax(0), volume_origin=vol_origin, voxel_size=resolution, volume_end=vol_end)


def run_fuse_instances(room_root, resolution):
    inst_vol = fuse_inconsistent_instances(room_root, resolution)
    npz = np.load(room_root / "sdf" / f"{resolution:.2f}.npz")
    sdf_vol, col_vol, sem_vol, vol_origin, vol_end = npz['sdf'], npz['color'], npz['semantics'], npz['volume_origin'], npz['volume_end']
    np.savez_compressed(room_root / "sdf" / f"{resolution:.2f}.npz", sdf=sdf_vol, color=col_vol, semantics=sem_vol.argmax(0), instances=inst_vol, volume_origin=vol_origin, voxel_size=resolution, volume_end=vol_end)


if __name__ == "__main__":
    _resolution = 0.05
    _max_depth = 5
    _root = Path("/home/yawarnihal/workspace/nerf-lightning/data/vfront/00154c06-2ee2-408a-9664-b8fd74742897/LivingRoom-17888/")
    # run_fuse_colors_and_semantics(_root, _resolution, _max_depth)
    run_fuse_instances(_root, _resolution)
