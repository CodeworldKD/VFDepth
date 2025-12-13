import os
import torch

_OPTIMIZER_NAME ='adam'


class BaseModel:
    def __init__(self, cfg):
        self._dataloaders = {}
        self.mode = None
        self.models = None
        self.optimizer = None
        self.lr_scheduler = None
        self.ddp_enable = False

    def read_config(self, cfg):
        raise NotImplementedError('Not implemented for BaseModel')

    def prepare_dataset(self):
        raise NotImplementedError('Not implemented for BaseModel')

    def set_optimizer(self):
        raise NotImplementedError('Not implemented for BaseModel')        
  
    def train_dataloader(self):
        return self._dataloaders['train']

    def val_dataloader(self):
        return self._dataloaders['val']

    def eval_dataloader(self):
        return self._dataloaders['eval']
    
    def set_train(self):
        self.mode = 'train'
        for m in self.models.values():
            m.train()

    def set_val(self):
        self.mode = 'val'
        for m in self.models.values():
            m.eval()
    
    def _format_model_filename(self, model_name, epoch=None, metric_name=None, metric_value=None):
        metric_name = metric_name.replace('_', '') if metric_name is not None else None

        parts = [model_name]
        if metric_name is not None and metric_value is not None:
            parts.append(f"{metric_name}_{metric_value:.3f}")
        if epoch is not None:
            parts.append(f"epoch{epoch}")
        return '_'.join(parts) + '.pth'
    
    def _save_models(self, target_dir, epoch=None, metric_name=None, metric_value=None):
        os.makedirs(target_dir, exist_ok=True)

        model_paths = {}
        for model_name, model in self.models.items():
            model_file_name = self._format_model_filename(model_name, epoch, metric_name, metric_value)
            model_file_path = os.path.join(target_dir, model_file_name)
            to_save = model.state_dict()
            torch.save(to_save, model_file_path)
            model_paths[model_name] = model_file_path
        return model_paths

    def save_model(self, epoch, metric_name=None, metric_value=None):
        curr_model_weights_dir = os.path.join(self.save_weights_root, f'weights_{epoch}')
        model_paths = self._save_models(curr_model_weights_dir, epoch, metric_name, metric_value)
        # save optimizer
        optim_file_path = os.path.join(curr_model_weights_dir, f'{_OPTIMIZER_NAME}.pth')
        torch.save(self.optimizer.state_dict(), optim_file_path)
        return model_paths

    def save_best_model(self, epoch, metric_name, metric_value, target_dir=None): # 这个返回的path没有 /weighs_epoch/
        save_dir = self.save_weights_root if target_dir is None else target_dir
        return self._save_models(save_dir, epoch, metric_name, metric_value)

    def load_weights(self):
        assert os.path.isdir(self.load_weights_dir), f'\tCannot find {self.load_weights_dir}'
        print(f'Loading a model from {self.load_weights_dir}')
        
        # to retrain
        if self.pretrain and self.ddp_enable:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % (self.world_size-1)}
            
        for n in self.models_to_load:
            print(f'Loading {n} weights...')
            path = os.path.join(self.load_weights_dir, f'{n}.pth')
            model_dict = self.models[n].state_dict()
            
            # distribute gpus for ddp retraining
            if self.pretrain and self.ddp_enable:
                pre_trained_dict = torch.load(path, map_location=map_location)
            else: 
                pre_trained_dict = torch.load(path)
                
            # load parameters
            pre_trained_dict = {k: v for k, v in pre_trained_dict.items() if k in model_dict}
            model_dict.update(pre_trained_dict)
            self.models[n].load_state_dict(model_dict)

        if self.mode == 'train':
            # loading adam state
            optim_file_path = os.path.join(self.load_weights_dir, f'{_OPTIMIZER_NAME}.pth')
            if os.path.isfile(optim_file_path):
                try:
                    print(f'Loading {_OPTIMIZER_NAME} weights')
                    optimizer_dict = torch.load(optim_file_path)
                    self.optimizer.load_state_dict(optimizer_dict)
                except ValueError:
                    print(f'\tCannnot load {_OPTIMIZER_NAME} - the optimizer will be randomly initialized')
            else:
                print(f'\tCannot find {_OPTIMIZER_NAME} weights, so the optimizer will be randomly initialized')