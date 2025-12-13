# Copyright (c) 2023 42dot. All rights reserved.
"""
针对没有真实 DDAD/NuScenes 数据的调试场景，提供一个可复现的合成数据集。
它会生成与 surround-view 训练流程兼容的张量字典，包括多尺度图像、
掩码、内外参以及简单的深度 GT，方便在 CPU/GPU 上打断点。
"""
import math
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _rotation_z(angle: float) -> torch.Tensor:
    """绕 z 轴的旋转矩阵。"""
    c = math.cos(angle)
    s = math.sin(angle)
    rot = torch.eye(3)
    rot[0, 0] = c
    rot[0, 1] = -s
    rot[1, 0] = s
    rot[1, 1] = c
    return rot


class SyntheticSurroundDataset(Dataset):
    """
    返回与 DDAD/NuScenes 数据格式一致的随机样本。

    每个样本都包括：
      * ('color', frame_id, scale) / ('color_aug', frame_id, scale)
      * 'mask', 'depth', 'extrinsics', 'K', 'inv_K'
      * 元信息 idx/dataset_idx/sensor_name/filename
    """

    def __init__(
        self,
        cfg: Dict,
        split: str,
        image_shape: Tuple[int, int],
        num_samples: int = 16,
        **_,  # 与真实构造函数保持一致，忽略多余参数
    ):
        self.split = split
        self.height, self.width = image_shape
        training_cfg = cfg['training']
        data_cfg = cfg['data']
        model_cfg = cfg['model']

        self.scales: List[int] = list(training_cfg['scales'])
        self.frame_ids: List[int] = list(training_cfg['frame_ids'])
        self.num_cams = data_cfg['num_cams']
        self.num_samples = data_cfg.get('synthetic_length', num_samples)
        self.seed = data_cfg.get('synthetic_seed', 42)
        self.radius = data_cfg.get('synthetic_radius', 4.0)
        self.height_offset = data_cfg.get('synthetic_height', 1.5)

        # 预计算基准内参
        base_K = torch.eye(4)
        base_K[0, 0] = self.width / 2.0
        base_K[1, 1] = self.height / 2.0
        base_K[0, 2] = self.width / 2.0
        base_K[1, 2] = self.height / 2.0
        base_K[2, 2] = 1.0
        self.base_K = base_K
        fusion_level = model_cfg.get('fusion_level', 0)
        # align_dataset 会准备到 level + 1
        self.k_scales = list(range(fusion_level + 2))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict:
        torch.manual_seed(self.seed + idx)

        sample: Dict = {}
        sample['idx'] = idx
        sample['dataset_idx'] = 0
        sample['sensor_name'] = 'synthetic'
        sample['filename'] = f'synth_{idx:06d}'

        base_rgb = torch.rand((self.num_cams, 3, self.height, self.width))
        base_noise = 0.05 * torch.randn_like(base_rgb)
        base_rgb_aug = torch.clamp(base_rgb + base_noise, 0.0, 1.0)

        # 训练用的 scale（默认只有 0）
        for scale in self.scales:
            h = self.height // (2 ** scale)
            w = self.width // (2 ** scale)
            sample[('color', 0, scale)] = F.interpolate(
                base_rgb, size=(h, w), mode='bilinear', align_corners=False)
            sample[('color_aug', 0, scale)] = F.interpolate(
                base_rgb_aug, size=(h, w), mode='bilinear', align_corners=False)

        # 时序上下文，只保留 scale=0
        for frame_id in self.frame_ids:
            if frame_id == 0:
                continue
            ctx = torch.rand((self.num_cams, 3, self.height, self.width))
            ctx_aug = torch.clamp(
                ctx + 0.03 * torch.randn_like(ctx), 0.0, 1.0)
            sample[('color', frame_id, 0)] = ctx
            sample[('color_aug', frame_id, 0)] = ctx_aug

        # 自遮挡 mask
        mask = torch.ones(
            (self.num_cams, 1, self.height, self.width), dtype=torch.float32)
        # 简单地给每个相机制造不同的遮挡块，方便可视化
        margin = self.width // 8
        for cam in range(self.num_cams):
            start = (cam * margin) % (self.width - margin)
            mask[cam, :, self.height // 4: self.height // 3,
                 start:start + margin] = 0.0
        sample['mask'] = mask

        # 生成简单的 GT 深度：带梯度的平面 + 不同相机的偏移
        yy = torch.linspace(0, 1, self.height).view(1, 1, self.height, 1)
        xx = torch.linspace(0, 1, self.width).view(1, 1, 1, self.width)
        depth_base = 10.0 + 40.0 * xx + 5.0 * torch.sin(2 * math.pi * yy)
        depth = depth_base.repeat(self.num_cams, 1, 1, 1)
        depth += torch.arange(self.num_cams).view(
            self.num_cams, 1, 1, 1) * 2.0
        sample['depth'] = depth

        # 相机外参：沿着车身周围平均分布
        extrinsics = torch.eye(4).unsqueeze(0).repeat(self.num_cams, 1, 1)
        for cam in range(self.num_cams):
            yaw = 2 * math.pi * cam / max(self.num_cams, 1)
            rot = _rotation_z(yaw)
            extrinsics[cam, :3, :3] = rot
            extrinsics[cam, 0, 3] = math.cos(yaw) * self.radius
            extrinsics[cam, 1, 3] = math.sin(yaw) * self.radius
            extrinsics[cam, 2, 3] = self.height_offset
        sample['extrinsics'] = extrinsics

        # 多尺度内参
        for scale in self.k_scales:
            scaled_K = self.base_K.clone()
            scaled_K[:2, :] /= (2 ** scale)
            K = scaled_K.unsqueeze(0).repeat(self.num_cams, 1, 1)
            sample[('K', scale)] = K
            sample[('inv_K', scale)] = torch.inverse(K)

        return sample
