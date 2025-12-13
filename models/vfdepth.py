# Copyright (c) 2023 42dot. All rights reserved.
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import construct_dataset
from network import *

from .base_model import BaseModel
from .geometry import Pose, ViewRendering
from .losses import DepthSynLoss, MultiCamLoss, SingleCamLoss

_NO_DEVICE_KEYS = ['idx', 'dataset_idx', 'sensor_name', 'filename']


class VFDepthAlgo(BaseModel):
    """
    Model class for "Self-supervised surround-view depth estimation with volumetric feature fusion"
    """
    def __init__(self, cfg, rank):
        super(VFDepthAlgo, self).__init__(cfg)
        self.rank = rank
        self.read_config(cfg)
        self.prepare_dataset(cfg, rank)
        self.models = self.prepare_model(cfg, rank)   
        self.losses = self.init_losses(cfg, rank)        
        self.view_rendering, self.pose = self.init_geometry(cfg, rank) 
        self.set_optimizer()
        
        if self.pretrain and rank == 0:
            self.load_weights()
        
    def read_config(self, cfg):    
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)
                
    def init_geometry(self, cfg, rank):
        view_rendering = ViewRendering(cfg, rank)
        pose = Pose(cfg)
        return view_rendering, pose
        
    def init_losses(self, cfg, rank):
        if self.aug_depth:
            loss_model = DepthSynLoss(cfg, rank)
        elif self.spatio_temporal or self.spatio:
            loss_model = MultiCamLoss(cfg, rank)
        else :
            loss_model = SingleCamLoss(cfg, rank)
        return loss_model
        
    def prepare_model(self, cfg, rank):
        models = {}
        models['pose_net'] = self.set_posenet(cfg)        
        models['depth_net'] = self.set_depthnet(cfg)

        # DDP training
        if self.ddp_enable == True:
            from torch.nn.parallel import DistributedDataParallel as DDP            
            process_group = dist.new_group(list(range(self.world_size)))
            # set ddp configuration
            for k, v in models.items():
                # sync batchnorm
                v = torch.nn.SyncBatchNorm.convert_sync_batchnorm(v, process_group)
                # DDP enable
                models[k] = DDP(v, device_ids=[rank], broadcast_buffers=True)
        return models

    def set_posenet(self, cfg):
        if self.pose_model =='fusion':
            return FusedPoseNet(cfg).cuda()
        else:
            return MonoPoseNet(cfg).cuda()    
        
    def set_depthnet(self, cfg):
        if self.depth_model == 'fusion':
            return FusedDepthNet(cfg).cuda()
        else:
            return MonoDepthNet(cfg).cuda()

    def prepare_dataset(self, cfg, rank):
        if rank == 0:
            print('### Preparing Datasets')
        
        if self.mode == 'train':
            self.set_train_dataloader(cfg, rank)
            if rank == 0 :
                self.set_val_dataloader(cfg)
                
        if self.mode == 'eval':
            self.set_eval_dataloader(cfg)

    def set_train_dataloader(self, cfg, rank):                 
        # jittering augmentation and image resizing for the training data
        _augmentation = {
            'image_shape': (int(self.height), int(self.width)), 
            'jittering': (0.2, 0.2, 0.2, 0.05),
            'crop_train_borders': (),
            'crop_eval_borders': ()
        }

        # construct train dataset
        train_dataset = construct_dataset(cfg, 'train', **_augmentation)

        dataloader_opts = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'pin_memory': True,
            'drop_last': True
        }

        if self.ddp_enable:
            dataloader_opts['shuffle'] = False
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, 
                num_replicas = self.world_size,
                rank=rank, 
                shuffle=True
            ) 
            dataloader_opts['sampler'] = self.train_sampler

        self._dataloaders['train'] = DataLoader(train_dataset, **dataloader_opts)
        num_train_samples = len(train_dataset)    
        self.num_total_steps = num_train_samples // (self.batch_size * self.world_size) * self.num_epochs

    def set_val_dataloader(self, cfg):         
        # Image resizing for the validation data
        _augmentation = {
            'image_shape': (int(self.height), int(self.width)),
            'jittering': (0.0, 0.0, 0.0, 0.0),
            'crop_train_borders': (),
            'crop_eval_borders': ()
        }

        # 构建验证数据集
        val_dataset = construct_dataset(cfg, 'val', **_augmentation)

        dataloader_opts = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': 0,
            'pin_memory': True,
            'drop_last': True
        }

        self._dataloaders['val']  = DataLoader(val_dataset, **dataloader_opts)
    
    def set_eval_dataloader(self, cfg):  
        # Image resizing for the validation data
        _augmentation = {
            'image_shape': (int(self.height), int(self.width)),
            'jittering': (0.0, 0.0, 0.0, 0.0),
            'crop_train_borders': (),
            'crop_eval_borders': ()
        }

        # construct validation dataset
        eval_dataset = construct_dataset(cfg, 'val', **_augmentation)

        dataloader_opts = {
            'batch_size': self.eval_batch_size,
            'shuffle': False,
            'num_workers': self.eval_num_workers,
            'pin_memory': True,
            'drop_last': True
        }

        self._dataloaders['eval'] = DataLoader(eval_dataset, **dataloader_opts)

    def set_optimizer(self):
        parameters_to_train = []
        for v in self.models.values():
            parameters_to_train += list(v.parameters())

        self.optimizer = optim.Adam(
        parameters_to_train, 
            self.learning_rate
        )

        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            self.scheduler_step_size,
            0.1
        )
    
    def process_batch(self, inputs, rank):
        """
        Pass a minibatch through the network and generate images, depth maps, and losses.
        """
        for key, ipt in inputs.items():
            if key not in _NO_DEVICE_KEYS:
                if 'context' in key:
                    inputs[key] = [ipt[k].float().to(rank) for k in range(len(inputs[key]))]
                else:
                    inputs[key] = ipt.float().to(rank)   

        outputs = self.estimate_vfdepth(inputs)
        losses = self.compute_losses(inputs, outputs)
        return outputs, losses  

    def estimate_vfdepth(self, inputs):
        """
        This function sets dataloader for validation in training.
        """          
        # pre-calculate inverse of the extrinsic matrix
        inputs['extrinsics_inv'] = torch.inverse(inputs['extrinsics']) # [b, N_cam, 4, 4]
        
        # init dictionary 
        outputs = {}
        for cam in range(self.num_cams):
            outputs[('cam', cam)] = {}

        pose_pred = self.predict_pose(inputs)                
        depth_feats = self.predict_depth(inputs)

        for cam in range(self.num_cams):       
            outputs[('cam', cam)].update(pose_pred[('cam', cam)])              
            outputs[('cam', cam)].update(depth_feats[('cam', cam)])

        if self.syn_visualize:
            outputs['disp_vis'] = depth_feats['disp_vis']
            
        self.compute_depth_maps(inputs, outputs) # 将网络预测的“归一化视差disp”转化为“真实物理深度Metric Depth”的关键后处理步骤
        return outputs

    def predict_pose(self, inputs):      
        """
        This function predicts poses.
        """          
        net = None
        if (self.mode != 'train') and self.ddp_enable:
            net = self.models['pose_net'].module
        else:
            net = self.models['pose_net']
        
        pose = self.pose.compute_pose(net, inputs)
        return pose

    def predict_depth(self, inputs):
        """
        This function predicts disparity maps.
        """                  
        net = None
        if (self.mode != 'train') and self.ddp_enable: 
            net = self.models['depth_net'].module
        else:
            net = self.models['depth_net']

        if self.depth_model == 'fusion':
            depth_feats = net(inputs)
        else:         
            depth_feats = {}
            for cam in range(self.num_cams):
                input_depth = inputs[('color_aug', 0, 0)][:, cam, ...]
                depth_feats[('cam', cam)] = net(input_depth)
        return depth_feats
    
    def compute_depth_maps(self, inputs, outputs):     
        """
        遍历所有相机和所有尺度，把网络输出的 disp全部转成 depth
        This function computes depth map for each viewpoint.
        """                  
        source_scale = 0
        for cam in range(self.num_cams):
            ref_K = inputs[('K', source_scale)][:, cam, ...] # 有四个内参是为了对应四个尺度的特征图 去了cam index的内参 形状[2, 4, 4]
            for scale in self.scales:
                disp = outputs[('cam', cam)][('disp', scale)]
                outputs[('cam', cam)][('depth', scale)] = self.to_depth(disp, ref_K)
                if self.aug_depth:
                    disp = outputs[('cam', cam)][('disp', scale, 'aug')]
                    outputs[('cam', cam)][('depth', scale, 'aug')] = self.to_depth(disp, ref_K)
    
    def to_depth(self, disp_in, K_in):        
        """
        范围映射 -> 上采样 -> 求倒数 -> 焦距缩放
        This function transforms disparity value into depth map while multiplying the value with the focal length.
        """
        min_disp = 1/self.max_depth
        max_disp = 1/self.min_depth
        disp_range = max_disp-min_disp

        disp_in = F.interpolate(disp_in, [self.height, self.width], mode='bilinear', align_corners=False)
        disp = min_disp + disp_range * disp_in
        depth = 1/disp # 转为深度，这里深度是 1.5-200 m
        return depth * K_in[:, 0:1, 0:1].unsqueeze(2)/self.focal_length_scale
        # 焦距缩放 K_in [2, 6, 4] K_in[:, 0:1, 0:1].unsqueeze(2)表示当前相机的焦距 f_x，规范焦距focal_length_scale = 300
        # 在针孔相机模型中，物体在图像上的大小 h 与距离 Z 和焦距 f 的关系是 h正比 f/Z。也就是说，Z正比于 f/h。预测的 disp 其实是“归一化视差”：网络只输出一个介于 [min_disp, max_disp] 的无量纲值，默认假设所有相机的焦距一致。如果直接 depth = 1 / disp，同一物体在不同焦距的摄像头下就会出现尺度不一致的问题。为了解决这个“不同相机内参导致的尺度漂移”，论文中提出了一个简单的“focal length scaling”（焦距归一化）
        # 如果网络只看图像（即只知道物体在图上的大小 h），它是无法区分“近处的小焦距拍摄”和“远处的大焦距拍摄”的。如果直接让网络预测真实深度 Z，对于不同焦距的相机（比如广角 vs 长焦），网络必须学会针对每个相机背诵不同的映射规则，这很难泛化。
        # 如果当前相机的实际焦距 fx 比参考焦距大，最终深度会按比例增大 乘上 fx/300；反之则减小。这样一来，网络只需要学会一个“统一的”从图像大小 h 到归一化视差 disp 的映射规则即可，不同相机间的尺度漂移问题也迎刃而解。就是把网络学到的“归一化视差”变回真实世界的绝对深度，同时自动适配不同摄像头的内参差异
    
    def compute_losses(self, inputs, outputs):
        """
        This function computes losses.
        """          
        losses = 0
        loss_fn = defaultdict(list)
        loss_mean = defaultdict(float)

        # generate image and compute loss per cameara
        for cam in range(self.num_cams):
            self.pred_cam_imgs(inputs, outputs, cam)
            cam_loss, loss_dict = self.losses(inputs, outputs, cam)
            
            losses += cam_loss  
            for k, v in loss_dict.items():
                loss_fn[k].append(v)

        losses /= self.num_cams
        
        for k in loss_fn.keys():
            loss_mean[k] = sum(loss_fn[k]) / float(len(loss_fn[k]))

        loss_mean['total_loss'] = losses        
        return loss_mean

    def pred_cam_imgs(self, inputs, outputs, cam):
        """
        This function renders projected images using camera parameters and depth information.
        """                  
        rel_pose_dict = self.pose.compute_relative_cam_poses(inputs, outputs, cam)
        self.view_rendering(inputs, outputs, cam, rel_pose_dict)  