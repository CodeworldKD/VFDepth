# Copyright (c) 2023 42dot. All rights reserved.
import time
from collections import defaultdict
from tqdm import tqdm
import os
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast

from utils import Logger


class VFDepthTrainer:
    """
    Trainer class for training and evaluation
    """
    def __init__(self, cfg, rank, use_tb=True):
        self.read_config(cfg)
        self.rank = rank        
        self.accumulation_steps = getattr(self, 'accumulation_steps', 2)
        self.use_grad_accumulation = getattr(self, 'use_grad_accumulation', False)
        self.use_amp = getattr(self, 'use_amp', False)
        self.scaler = GradScaler(enabled=self.use_amp)
        if rank == 0:
            self.logger = Logger(cfg, use_tb)
            self.depth_metric_names = self.logger.get_metric_names()
            self.best_metric = {'abs_rel': float('inf'), 'rms': float('inf'), 'a1': 0.0}
            self.best_paths = {'abs_rel': {}, 'rms': {}, 'a1': {}}

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
                # val_metric = self.validate(model)
                # metric_value = val_metric.get('abs_rel') if val_metric is not None else None
                # model.save_model(self.epoch, 'abs_rel', metric_value) # 这里可以改指标来保存
                model.save_model(self.epoch)
                print(f'---------- 保存 {self.epoch} epoch模型-------------')
                
            if self.ddp_enable:
                dist.barrier()
                
        if self.rank == 0:
            self.logger.close_tb()
        
    def train(self, model, data_loader, start_time):
        """
        This function trains models.
        """
        model.set_train()
        accumulation_steps = max(1, self.accumulation_steps)
        use_accum = self.use_grad_accumulation and accumulation_steps > 1
        scaler = self.scaler
        total_batches = len(data_loader)
        if use_accum:
            model.optimizer.zero_grad(set_to_none=True)
        process_bar = tqdm(data_loader)
        for batch_idx, inputs in enumerate(process_bar):      
            before_op_time = time.time()
            if not use_accum:
                model.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                outputs, losses = model.process_batch(inputs, self.rank)
                loss = losses['total_loss']
                if use_accum:
                    loss = loss/accumulation_steps

            if self.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if use_accum:
                should_step = ((batch_idx + 1) % accumulation_steps == 0) or ((batch_idx + 1) == total_batches)
                if should_step:
                    if self.use_amp:
                        scaler.step(model.optimizer)
                        scaler.update()
                    else:
                        model.optimizer.step()
                    model.optimizer.zero_grad(set_to_none=True)
            else:
                if self.use_amp:
                    scaler.step(model.optimizer)
                    scaler.update()
                else:
                    model.optimizer.step()

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
        只跑了验证集的一个 batch
        """
        model.set_val()
        inputs = next(self.val_iter)
            
        outputs, losses = model.process_batch(inputs, self.rank)
        
        if 'depth' in inputs:
            depth_eval_metric, depth_eval_median = self.logger.compute_depth_losses(inputs, outputs, vis_scale=True)
            self.logger.print_perf(depth_eval_metric, 'metric')
            self.logger.print_perf(depth_eval_median, 'median')
            self._update_best_model(model, depth_eval_metric)

        self.logger.log_tb('val', inputs, outputs, losses, self.step)            
        del inputs, outputs, losses
        
        model.set_train()
        return depth_eval_metric
        
    def _update_best_model(self, model, metric_dict):
        # 如果log的时候指标变好，保存新的模型权重
        abs_rel = metric_dict['abs_rel']
        if abs_rel < self.best_metric['abs_rel']:
            self._remove_old_best('abs_rel')
            self.best_paths['abs_rel'] = model.save_best_model(self.epoch, 'abs_rel', abs_rel)
            self.best_metric['abs_rel'] = abs_rel
            print(f"更新全局最优的模型，其中abs_rel为{self.best_metric['abs_rel']}")

        # rmse = metric_dict['rms']
        # if rmse < self.best_metric['rms']:
        #     self._remove_old_best('rms')
        #     self.best_paths['rms'] = model.save_best_model(self.epoch, 'rms', rmse)
        #     self.best_metric['rms'] = rmse

        # a1 = metric_dict['a1']
        # if a1 > self.best_metric['a1']:
        #     self._remove_old_best('a1')
        #     self.best_paths['a1'] = model.save_best_model(self.epoch, 'a1', a1)
        #     self.best_metric['a1'] = a1

    def _remove_old_best(self, metric_name):
        old_paths = self.best_paths.get(metric_name, {})
        for _, path in old_paths.items():
            if os.path.exists(path):
                os.remove(path)

    @torch.no_grad()
    def evaluate(self, model, vis_results=False):
        """
        此函数用于在完整的验证数据集上评估模型
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
