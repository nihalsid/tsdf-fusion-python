import math
import pickle
import time
from pathlib import Path
import random

import numpy as np
import scipy.optimize
import torch
import torch.nn.functional as F
from skimage import measure
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

from util.distinct_colors import DistinctColors
from util.front_to_nyu_subset import selected_nyu_fg_classes, front_to_nyu
from util.misc import pcwrite, meshwrite
from util.ray import get_ray_directions_with_intrinsics, get_rays

loss_instances_cluster = None


class VFrontSDFDataset(Dataset):

    def __init__(self, root_dir, image_dim, semantic_classes=selected_nyu_fg_classes, max_rays=1024, instance_dir='filtered_asset_inc', instance_to_semantic_key='instance_to_semantic_inc'):
        self.root_dir = root_dir
        self.image_dim = image_dim
        self.instance_directory = instance_dir
        self.instance_to_semantic_key = instance_to_semantic_key
        self.cam2scenes = {}
        self.world2scene = None
        self.scene2normscene = None
        self.intrinsics = {}
        self.semantic_classes = semantic_classes
        self.cam2normscene = {}
        self.normscene_scale = None
        self.all_rays = []
        self.all_semantics = []
        self.all_instances = []
        self.segmentation_data = None
        self.all_frame_names = []
        self.indices = []
        self.all_origins = []
        self.world2scene = np.eye(4, dtype=np.float32)
        tsdf_data = np.load(root_dir / "sdf" / "0.05.npz")
        self.scene2normscene = torch.tensor([
            [1 / tsdf_data['voxel_size'], 0, 0, -tsdf_data['volume_origin'][0] / tsdf_data['voxel_size']],
            [0, 1 / tsdf_data['voxel_size'], 0, -tsdf_data['volume_origin'][1] / tsdf_data['voxel_size']],
            [0, 0, 1 / tsdf_data['voxel_size'], -tsdf_data['volume_origin'][2] / tsdf_data['voxel_size']],
            [0, 0, 0, 1]
        ]).float()
        self.voxel_size = tsdf_data['voxel_size']
        self.volume_origin = tsdf_data['volume_origin']
        self.volume_end = tsdf_data['volume_end']
        self.sdf_grid = torch.from_numpy(tsdf_data['sdf']).float().unsqueeze(0)
        self.near = 0.01
        self.far = max(self.sdf_grid.shape)
        self.setup_data()
        all_rays = []
        all_instances = []
        for i in tqdm(range(len(self.indices)), desc='loading dataset'):
            all_semantics_view = self.all_semantics[self.all_origins == self.indices[i]]
            all_instances_view = self.all_instances[self.all_origins == self.indices[i]]
            all_rays_view = self.all_rays[self.all_origins == self.indices[i], :]
            mask = torch.zeros_like(all_semantics_view).bool()
            for semantic_class in semantic_classes:
                mask = torch.logical_or(mask, all_semantics_view == semantic_class)
            if mask.sum() > 0:
                all_rays.append(all_rays_view[mask, :])
                all_instances.append(all_instances_view[mask])
        self.all_rays = all_rays
        self.all_instances = all_instances
        self.max_rays = max_rays

    def move_buffers_to_device(self, device):
        self.sdf_grid = self.sdf_grid.to(device)

    def setup_data(self):
        self.all_frame_names = sorted([x.stem for x in (self.root_dir / "rgb").iterdir() if x.name.endswith('.png')])
        self.indices = list(range(len(self.all_frame_names)))
        dims, intrinsics, cam2scene = [], [], []
        img_h, img_w = np.array(Image.open(self.root_dir / "rgb" / f"{self.all_frame_names[0]}.png")).shape[:2]
        for sample_index in self.indices:
            sample_annotation = pickle.load(open(self.root_dir / 'annotation' / f'{self.all_frame_names[sample_index]}.pkl', 'rb'))
            intrinsic = np.array(sample_annotation['cam_K'], dtype=np.float32)
            cam2world = np.array(sample_annotation['cam2world_matrix'], dtype=np.float32) @ np.diag([1, -1, -1, 1])
            cam2scene.append(torch.from_numpy(self.world2scene @ cam2world).float())
            self.cam2scenes[sample_index] = cam2scene[-1]
            dims.append([img_h, img_w])
            intrinsics.append(torch.from_numpy(intrinsic))
            self.intrinsics[sample_index] = intrinsic
            self.intrinsics[sample_index] = torch.from_numpy(np.diag([self.image_dim[1] / img_w, self.image_dim[0] / img_h, 1]) @ self.intrinsics[sample_index]).float()

        self.normscene_scale = self.scene2normscene[0, 0]

        for sample_index in self.indices:
            self.cam2normscene[sample_index] = self.scene2normscene @ self.cam2scenes[sample_index]

        for sample_index in self.indices:
            rays, semantics, instances, room_mask = self.load_sample(sample_index)
            self.all_rays.append(rays[room_mask, :])
            self.all_semantics.append(semantics[room_mask])
            self.all_instances.append(instances[room_mask])
            self.all_origins.append(torch.ones_like(semantics[room_mask]) * sample_index)
        self.all_rays = torch.cat(self.all_rays, 0)
        self.all_semantics = torch.cat(self.all_semantics, 0)
        self.all_instances = torch.cat(self.all_instances, 0)
        self.all_origins = torch.cat(self.all_origins, 0)

    def load_sample(self, sample_index):
        cam2normscene = self.cam2normscene[sample_index]
        semantics = Image.open(self.root_dir / "sem" / f"{self.all_frame_names[sample_index]}.png")
        instances = Image.open(self.root_dir / self.instance_directory / f"{self.all_frame_names[sample_index]}.png")
        room_mask = Image.open(self.root_dir / "room_mask" / f"{self.all_frame_names[sample_index]}.png")
        # noinspection PyTypeChecker
        semantics = front_to_nyu(np.array(semantics.resize(self.image_dim[::-1], Image.NEAREST)))
        semantics = torch.from_numpy(semantics).long()
        # noinspection PyTypeChecker
        instances = torch.from_numpy(np.array(instances.resize(self.image_dim[::-1], Image.NEAREST))).long()
        # noinspection PyTypeChecker
        room_mask = torch.from_numpy(np.array(room_mask.resize(self.image_dim[::-1], Image.NEAREST))).bool()

        directions = get_ray_directions_with_intrinsics(self.image_dim[0], self.image_dim[1], self.intrinsics[sample_index].numpy())
        rays_o, rays_d = get_rays(directions, cam2normscene)

        rays = torch.cat(
            [rays_o, rays_d, self.near * torch.ones_like(rays_o[:, :1]), self.far * torch.ones_like(rays_o[:, :1]), ], 1,
        )
        return rays, semantics.reshape(-1), instances.reshape(-1), room_mask.reshape(-1)

    def __getitem__(self, idx):
        selected_rays = self.all_rays[idx]
        selected_instances = self.all_instances[idx]
        if selected_rays.shape[0] > self.max_rays:
            sampled_indices = random.sample(range(selected_rays.shape[0]), self.max_rays)
            selected_rays = selected_rays[sampled_indices, :]
            selected_instances = selected_instances[sampled_indices]
        sample = {
            f"rays": selected_rays,
            f"instances": selected_instances,
        }
        return sample

    def __len__(self):
        return len(self.all_rays)


class NeuSFixedGridRenderer(torch.nn.Module):

    def __init__(self, grid_dimensions, raymarch_weight_thres=0.0001):
        super().__init__()
        self.register_buffer("grid_dimensions", torch.LongTensor(grid_dimensions))
        self.n_samples = 0
        self.step_size = 0
        self.step_ratio = 0.25
        self.raymarch_weight_thres = raymarch_weight_thres
        self.update_step_size()

    def sample_points_in_box(self, rays):
        rays_o, rays_d, nears, fars = rays[:, 0:3], rays[:, 3:6], rays[:, 6], rays[:, 7]
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.grid_dimensions - rays_o) / vec
        rate_b = - rays_o / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=nears, max=fars)
        rng = torch.arange(self.n_samples)[None].float()
        step = self.step_size * rng.to(rays_o.device)
        interpx = (t_min[..., None] + step)
        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((torch.zeros_like(self.grid_dimensions) > rays_pts) | (rays_pts > self.grid_dimensions)).any(dim=-1)
        rays_pts = rays_pts[:, :, [2, 1, 0]]
        return rays_pts, interpx, ~mask_outbbox

    def update_step_size(self):
        box_diag = torch.sqrt(torch.sum(torch.square(self.grid_dimensions)))
        self.n_samples = int((box_diag / self.step_ratio).item()) + 1
        self.step_size = (box_diag / self.n_samples).mean()

    @staticmethod
    def raw_to_alpha(sdf, dists, inv_s):
        n_rays, n_samples = sdf.shape
        sdf = sdf.reshape(-1, 1)
        dists = dists.reshape(-1, 1)
        iter_cos = -1  # always non-positive
        estimated_next_sdf = sdf + iter_cos * dists * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists * 0.5
        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
        p = prev_cdf - next_cdf
        c = prev_cdf
        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0).reshape(n_rays, n_samples)
        T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)
        weights = alpha * T[:, :-1]
        return alpha, weights, T[:, -1:]

    def forward(self, sdf_grid, instance_grid, rays, inv_s):
        zyx_sampled, z_vals, mask_zyx = self.sample_points_in_box(rays)
        dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        sdf = torch.ones(zyx_sampled.shape[:-1], device=zyx_sampled.device) * float('inf')
        instances = torch.zeros((*zyx_sampled.shape[:2], instance_grid.shape[1]), device=zyx_sampled.device)

        grid_zyx_sampled = (zyx_sampled / self.grid_dimensions[None, None, [2, 1, 0]]) * 2 - 1
        if mask_zyx.any():
            sdf[mask_zyx] = F.grid_sample(sdf_grid, grid_zyx_sampled[mask_zyx].unsqueeze(0).unsqueeze(0).unsqueeze(0), align_corners=True, padding_mode="border").squeeze()

        alpha, weight, bg_weight = self.raw_to_alpha(sdf, dists, inv_s)
        appearance_mask = weight > self.raymarch_weight_thres

        if appearance_mask.any():
            instances[appearance_mask] = F.grid_sample(instance_grid, grid_zyx_sampled[appearance_mask].unsqueeze(0).unsqueeze(0).unsqueeze(0), align_corners=True,
                                                       padding_mode="border").squeeze().T

        instances_map = torch.sum(weight[..., None] * instances, -2)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)

        return instances_map, depth_map


def calculate_instance_clustering_loss(labels_gt, instance_features):
    virtual_gt_labels = create_virtual_gt_with_linear_assignment(labels_gt, instance_features)
    predicted_labels = instance_features.argmax(dim=-1)
    if torch.any(virtual_gt_labels != predicted_labels):  # should never reinforce correct labels
        return loss_instances_cluster(instance_features, virtual_gt_labels)
    return torch.tensor(0., device=labels_gt.device, requires_grad=True)


@torch.no_grad()
def create_virtual_gt_with_linear_assignment(labels_gt, predicted_scores):
    labels = sorted(torch.unique(labels_gt).cpu().tolist())[:predicted_scores.shape[-1]]
    predicted_probabilities = torch.softmax(predicted_scores, dim=-1)
    cost_matrix = np.zeros([len(labels), predicted_probabilities.shape[-1]])
    for lidx, label in enumerate(labels):
        cost_matrix[lidx, :] = -(predicted_probabilities[labels_gt == label, :].sum(dim=0) / ((labels_gt == label).sum() + 1e-4)).cpu().numpy()
    assignment = scipy.optimize.linear_sum_assignment(np.nan_to_num(cost_matrix))
    new_labels = torch.zeros_like(labels_gt)
    for aidx, lidx in enumerate(assignment[0]):
        new_labels[labels_gt == labels[lidx]] = assignment[1][aidx]
    return new_labels


def fuse_inconsistent_instances(root_dataset_path, resolution):
    def get_inv_s(_epoch):
        return 25 - 5 * math.cos(math.pi * _epoch / max_epochs)

    global loss_instances_cluster
    print_interval = 100
    max_instances = 50
    max_epochs = 50
    device = torch.device("cuda:0")
    dataset = VFrontSDFDataset(root_dataset_path, (256, 256))
    dataset.move_buffers_to_device(device)
    fused_instances = torch.zeros([1, max_instances, dataset.sdf_grid.shape[2], dataset.sdf_grid.shape[3], dataset.sdf_grid.shape[4]], requires_grad=True, dtype=torch.float32, device=device)
    renderer = NeuSFixedGridRenderer([dataset.sdf_grid.shape[2], dataset.sdf_grid.shape[3], dataset.sdf_grid.shape[4]]).to(device)
    loss_instances_cluster = torch.nn.CrossEntropyLoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam([fused_instances], lr=5e-3)
    for epoch in range(max_epochs):
        cumulative_loss = 0
        elapsed_time = 0
        for sample_idx in range(len(dataset)):
            t0 = time.time()
            optimizer.zero_grad(set_to_none=True)
            sample = dataset[sample_idx]
            rays, instances = sample['rays'].to(device), sample['instances'].to(device)
            out_instances, _ = renderer(dataset.sdf_grid, fused_instances, rays, get_inv_s(epoch))
            loss = calculate_instance_clustering_loss(instances, out_instances)
            cumulative_loss += loss.item()
            loss.backward()
            optimizer.step()
            elapsed_time = elapsed_time + time.time() - t0
            if (sample_idx + 1) % print_interval == 0:
                print(f'E{epoch:03d}[{sample_idx:05d}]: {cumulative_loss / print_interval:.4f} with inv_s={get_inv_s(epoch):.2f} at {print_interval / elapsed_time:.2f} it/s')
                cumulative_loss = 0
                elapsed_time = 0

    distinct_colors = DistinctColors()
    instances = torch.cat([torch.ones_like(dataset.sdf_grid) * -float('inf'), fused_instances.detach()], dim=1)
    instances = instances.cpu().squeeze(0).permute((1, 2, 3, 0)).numpy()
    fused_semantics = torch.from_numpy(np.load(root_dataset_path / "sdf" / f"{resolution:.2f}.npz")["semantics"])
    grid_background_mask = torch.ones([dataset.sdf_grid.shape[2], dataset.sdf_grid.shape[3], dataset.sdf_grid.shape[4]]).bool()
    for c in dataset.semantic_classes:
        grid_background_mask = torch.logical_and(grid_background_mask, fused_semantics != c)
    grid_background_mask = torch.logical_or(grid_background_mask, torch.logical_or(dataset.sdf_grid[0, 0].cpu() >= 1, dataset.sdf_grid[0, 0].cpu() <= (-0.75)))
    instances[grid_background_mask, 0] = float('inf')

    # Marching cubes
    verts, faces, norms, vals = measure.marching_cubes(dataset.sdf_grid[0, 0].cpu().numpy(), level=0)
    verts_ind = np.round(verts).astype(int)
    verts_vox = verts
    verts_scaled = verts * dataset.voxel_size + dataset.volume_origin

    instance_vals = instances[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2], :]
    instance_vals = instance_vals.argmax(-1)
    instances_colored = (distinct_colors.get_color_fast_numpy(instance_vals) * 255).astype(np.uint8)

    pc = np.hstack([verts_scaled, instances_colored])
    (root_dataset_path / "sdf" / "visualization").mkdir(exist_ok=True, parents=True)
    pcwrite(root_dataset_path / "sdf" / "visualization" / "inst_pc.ply", pc)

    meshwrite(root_dataset_path / "sdf" / "visualization" / "inst.ply", verts_scaled, faces, norms, instances_colored)
    meshwrite(root_dataset_path / "sdf" / "visualization" / "inst_vox_space.ply", verts_vox, faces, norms, instances_colored)

    return instance_vals


if __name__ == "__main__":
    fuse_inconsistent_instances(Path("/home/yawarnihal/workspace/nerf-lightning/data/vfront/00154c06-2ee2-408a-9664-b8fd74742897/LivingRoom-17888/"), 0.05)
