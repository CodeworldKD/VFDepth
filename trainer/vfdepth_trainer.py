# Copyright (c) 2023 42dot. All rights reserved.
import time
from collections import defaultdict
from tqdm import tqdm
import os
import torch
import torch.distributed as dist

from utils import Logger


class VFDepthTrainer:
    """
    Trainer class for training and evaluation
    """
    def __init__(self, cfg, rank, use_tb=True):
        self.read_config(cfg)
        self.rank = rank        
        if rank == 0:
            self.logger = Logger(cfg, use_tb)
            self.depth_metric_names = self.logger.get_metric_names()

    def read_config(self, cfg):
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)

    def learn(self, model):
        """
        This function sets training process.
        """        
        train_dataloader = model.train_dataloader()
        if self.rank == 0:
            val_dataloader = model.val_dataloader()
            self.val_iter = iter(val_dataloader)
        
        self.step = 0
        start_time = time.time()
        for self.epoch in range(self.num_epochs):
            if self.ddp_enable:
                model.train_sampler.set_epoch(self.epoch)     

            self.train(model, train_dataloader, start_time)
            
            # save model after each epoch using rank 0 gpu 
            if self.rank == 0:
                model.save_model(self.epoch) # 保存 epoch 结束时的权重
                print('-'*110) 
                
            if self.ddp_enable:
                dist.barrier()
                
        if self.rank == 0:
            self.logger.close_tb()
        
    def train(self, model, data_loader, start_time):
        """
        This function trains models.
        """
        model.set_train()
        # --- 【适配显存修改点 1】设置累积步数 ---
        accumulation_steps = 2 
        
        # 初始化梯度
        model.optimizer.zero_grad(set_to_none=True)
        process_bar = tqdm(data_loader)
        for batch_idx, inputs in enumerate(process_bar):      
            before_op_time = time.time()
            # --- 【适配显存修改点 2】不要在这里清空梯度，移到下面去 ---
            # model.optimizer.zero_grad(set_to_none=True)
            outputs, losses = model.process_batch(inputs, self.rank)
            # losses['total_loss'].backward()
            # 累积梯度
            loss = losses['total_loss']/accumulation_steps
            loss.backward()
            # model.optimizer.step()
            # 每隔 accumulation_steps 更新  
            if (batch_idx + 1) % accumulation_steps == 0:
                model.optimizer.step()
                model.optimizer.zero_grad(set_to_none=True)

            if self.rank == 0: 
                self.logger.update(
                    'train', 
                    self.epoch, 
                    self.world_size,
                    batch_idx, 
                    self.step,
                    start_time,
                    before_op_time, 
                    inputs,
                    outputs,
                    losses
                )

                if self.logger.is_checkpoint(self.step): # 根据step打印结果
                    self.validate(model)

            if self.ddp_enable:
                dist.barrier()

            self.step += 1

        model.lr_scheduler.step()
        
    @torch.no_grad()
    def validate(self, model):
        """
        This function validates models on validation dataset to monitor training process.
        """
        model.set_val()
        inputs = next(self.val_iter)
            
        outputs, losses = model.process_batch(inputs, self.rank)
        
        if 'depth' in inputs:
            depth_eval_metric, depth_eval_median = self.logger.compute_depth_losses(inputs, outputs, vis_scale=True)
            self.logger.print_perf(depth_eval_metric, 'metric')
            self.logger.print_perf(depth_eval_median, 'median')

        self.logger.log_tb('val', inputs, outputs, losses, self.step)            
        del inputs, outputs, losses
        
        model.set_train()
        
    @torch.no_grad()
    def evaluate(self, model, vis_results=False):
        """
        This function evaluates models on full validation dataset.
        """
        eval_dataloader = model.eval_dataloader()
        
        # load model
        model.load_weights()
        model.set_val()
        
        avg_depth_eval_metric = defaultdict(float)
        avg_depth_eval_median = defaultdict(float)        
        
        process = tqdm(eval_dataloader)
        for batch_idx, inputs in enumerate(process):   
            # visualize synthesized depth maps
            if self.syn_visualize and batch_idx < self.syn_idx:
                continue
                
            outputs, _ = model.process_batch(inputs, self.rank)
            depth_eval_metric, depth_eval_median = self.logger.compute_depth_losses(inputs, outputs)
            
            for key in self.depth_metric_names:
                avg_depth_eval_metric[key] += depth_eval_metric[key]
                avg_depth_eval_median[key] += depth_eval_median[key]
            
            if vis_results:
                self.logger.log_result(inputs, outputs, batch_idx, self.syn_visualize)
            
            if self.syn_visualize and batch_idx >= self.syn_idx:
                process.close()
                break
 
        for key in self.depth_metric_names:
            avg_depth_eval_metric[key] /= len(eval_dataloader)
            avg_depth_eval_median[key] /= len(eval_dataloader)

        print('Evaluation result...\n')
        self.logger.print_perf(avg_depth_eval_metric, 'metric')
        self.logger.print_perf(avg_depth_eval_median, 'median')
