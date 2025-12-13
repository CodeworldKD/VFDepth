# Copyright (c) 2023 42dot. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import conv2d, pack_cam_feat, unpack_cam_feat
from .volumetric_fusionnet import VFNet

from external.layers import ResnetEncoder, PoseDecoder


class FusedPoseNet(nn.Module):
    """
    Canonical motion estimation module.
    """
    def __init__(self, cfg):
        super(FusedPoseNet, self).__init__()
        self.read_config(cfg)

        # feature encoder        
        # resnet feat: 64(1/2), 64(1/4), 128(1/8), 256(1/16), 512(1/32)
        self.encoder = ResnetEncoder(self.num_layers, self.weights_init, 2) # number of layers, pretrained, number of input images
        del self.encoder.encoder.fc
        enc_feat_dim = sum(self.encoder.num_ch_enc[self.fusion_level:])
        self.conv1x1 = conv2d(enc_feat_dim, self.fusion_feat_in_dim, kernel_size=1, padding_mode='reflect')
        
        # fusion net
        fusion_feat_out_dim = self.encoder.num_ch_enc[self.fusion_level]
        self.fusion_net = VFNet(cfg, self.fusion_feat_in_dim, fusion_feat_out_dim, model = 'pose')

        # depth decoder
        self.pose_decoder = PoseDecoder(num_ch_enc = [fusion_feat_out_dim],
                                        num_input_features=1, 
                                        num_frames_to_predict_for=1, 
                                        stride=2)
    
    def read_config(self, cfg):
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)
                
    def forward(self, inputs, frame_ids, _):
        outputs = {}
    
        # initialize dictionary
        for cam in range(self.num_cams):
            outputs[('cam', cam)] = {}

        lev = self.fusion_level # 2

        # packed images for surrounding view
        # 一次输入包括前后时序的6*3=18张图片，如果batchsiez = 2 总共36张图片
        cur_image = inputs[('color_aug', frame_ids[0], 0)]  # [b,6,3,h,w] 当前帧
        next_image = inputs[('color_aug', frame_ids[1], 0)] # [b,6,3,h,w] 下一帧
        
        pose_images = torch.cat([cur_image, next_image], 2) # [b,6,6,h,w] 把时间维度的图片拼接在channel维度上
        packed_pose_images = pack_cam_feat(pose_images) # [b*6, 6, h, w] 打包 Batch 和 Camera 维度
        
        packed_feats = self.encoder(packed_pose_images) # 图像编码器
        
        # aggregate feature H / 2^(lev+1) x W / 2^(lev+1)
        _, _, up_h, up_w = packed_feats[lev].size()
        
        # packed_feats[lev:lev+1]   [b*6,128,h/8,w/8]   128+256+512=896
        packed_feats_list = packed_feats[lev:lev+1] \
                        + [F.interpolate(feat, [up_h, up_w], mode='bilinear', align_corners=True) for feat in packed_feats[lev+1:]]           

        packed_feats_agg = self.conv1x1(torch.cat(packed_feats_list, dim=1))       # [b*6,256,h/8,w/8]   
        feats_agg = unpack_cam_feat(packed_feats_agg, self.batch_size, self.num_cams) # [b, N_cam, 256, h/8, w/8]
    
        # fusion_net, backproject each feature into the 3D voxel space
        bev_feat = self.fusion_net(inputs, feats_agg)        # [b,128,25,25]
        axis_angle, translation = self.pose_decoder([[bev_feat]]) # [b, 1, 1, 3] [b, 1, 1, 3]
        return axis_angle, torch.clamp(translation, -4.0, 4.0) # for DDAD dataset
        # clamp强制将预测出的相机平移量的每一个分量（x, y, z）都限制在 [-4.0, 4.0] 之间