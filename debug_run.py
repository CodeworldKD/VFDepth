#!/usr/bin/env python3
"""
简化版的 VFDepth 训练入口，用于调试/断点。
会加载配置、构造模型与数据集（可选 synthetic），
然后按指定步数运行 process_batch，方便在 VSCode 中逐行单步。
"""
import argparse

import torch

import utils
from models import VFDepthAlgo


def parse_args():
    parser = argparse.ArgumentParser(description='VFDepth debugging runner')
    parser.add_argument('--config_file', type=str,
                        default='./configs/debug/ddad_surround_fusion_synth.yaml',
                        help='Config yaml file（建议使用 synthetic 版本）')
    parser.add_argument('--mode', choices=['train', 'val'], default='train',
                        help='选择加载 train loader 还是 val loader')
    parser.add_argument('--steps', type=int, default=1,
                        help='执行多少个 batch 后停止，方便断点')
    parser.add_argument('--synthetic', action='store_true',
                        help='强制覆盖配置，改用 synthetic 数据集')
    parser.add_argument('--device', type=int, default=0,
                        help='传给模型的 rank（0 表示 cuda:0，如果无 GPU 可在 VSCode 里改为 cpu）')
    return parser.parse_args()


def fetch_loader(model: VFDepthAlgo, mode: str):
    if mode == 'train':
        model.set_train()
        return model.train_dataloader()
    else:
        model.set_val()
        return model.val_dataloader()


def main():
    args = parse_args()
    cfg = utils.get_config(args.config_file, mode=args.mode)
    if args.synthetic:
        cfg['data']['dataset'] = 'synthetic'

    rank = args.device
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    model = VFDepthAlgo(cfg, rank=rank)
    data_loader = fetch_loader(model, args.mode)

    for step, inputs in enumerate(data_loader):
        outputs, losses = model.process_batch(inputs, rank)
        loss_val = losses['total_loss'].item()
        print(f'[debug_run] step={step} loss={loss_val:.4f} '
              f'cams={len(outputs)} keys={list(losses.keys())}')
        if step + 1 >= args.steps:
            break


if __name__ == '__main__':
    main()
