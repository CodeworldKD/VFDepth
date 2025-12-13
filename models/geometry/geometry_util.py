# Copyright (c) 2023 42dot. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from pytorch3d.transforms import axis_angle_to_matrix 

        
def vec_to_matrix(rot_angle, trans_vec, invert=False):
    """
    This function transforms rotation angle and translation vector into 4x4 matrix.
    将 6自由度(6-DoF)的位姿向量(3个旋转参数 + 3个平移参数)转换成标准的 4*4 齐次变换矩阵
    """
    # initialize matrices
    b, _, _ = rot_angle.shape # [b,1,3]
    R_mat = torch.eye(4).repeat([b, 1, 1]).to(device=rot_angle.device)
    T_mat = torch.eye(4).repeat([b, 1, 1]).to(device=rot_angle.device)

    R_mat[:, :3, :3] = axis_angle_to_matrix(rot_angle).squeeze(1) # [b,4,4]    (b, 1, 3, 3).squeeze(1) 去掉第一维 --> [b, 3, 3]
    t_vec = trans_vec.clone().contiguous().view(-1, 3, 1) # [b,3,1]

    if invert == True:
        R_mat = R_mat.transpose(1,2)
        t_vec = -1 * t_vec

    T_mat[:, :3,  3:] = t_vec  # [b,4,4]

    if invert == True:
        P_mat = torch.matmul(R_mat, T_mat)
    else :
        P_mat = torch.matmul(T_mat, R_mat)  # 如果你有一个点 𝑥，想先对它进行旋转，再进行平移：x' = T * R * x
    return P_mat   # [b,4,4]


class Projection(nn.Module):
    """
    This class computes projection and reprojection function. 
    """
    def __init__(self, batch_size, height, width, device):
        super().__init__()
        self.batch_size = batch_size
        self.width = width
        self.height = height
        
        # initialize img point grid
        img_points = np.meshgrid(range(width), range(height), indexing='xy')
        img_points = torch.from_numpy(np.stack(img_points, 0)).float()
        img_points = torch.stack([img_points[0].view(-1), img_points[1].view(-1)], 0).repeat(batch_size, 1, 1)
        img_points = img_points.to(device)
        
        self.to_homo = torch.ones([batch_size, 1, width*height]).to(device)
        self.homo_points = torch.cat([img_points, self.to_homo], 1)

    def backproject(self, invK, depth):
        """
        This function back-projects 2D image points to 3D.
        """
        depth = depth.view(self.batch_size, 1, -1)

        points3D = torch.matmul(invK[:, :3, :3], self.homo_points)
        points3D = depth*points3D
        return torch.cat([points3D, self.to_homo], 1)
    
    def reproject(self, K, points3D, T):
        """
        This function reprojects transformed 3D points to 2D image coordinate.
        """
        # project points 
        points2D = (K @ T)[:,:3, :] @ points3D

        # normalize projected points for grid sample function
        norm_points2D = points2D[:, :2, :]/(points2D[:, 2:, :] + 1e-7)
        norm_points2D = norm_points2D.view(self.batch_size, 2, self.height, self.width)
        norm_points2D = norm_points2D.permute(0, 2, 3, 1)

        norm_points2D[..., 0 ] /= self.width - 1
        norm_points2D[..., 1 ] /= self.height - 1
        norm_points2D = (norm_points2D-0.5)*2
        return norm_points2D        

    def forward(self, depth, T, bp_invK, rp_K):
        cam_points = self.backproject(bp_invK, depth)
        pix_coords = self.reproject(rp_K, cam_points, T)
        return pix_coords