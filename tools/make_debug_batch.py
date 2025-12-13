#!/usr/bin/env python3
"""
Utility to dump a single VFDepth batch for interactive debugging.

Example:
    python tools/make_debug_batch.py \\
        --config_file ./configs/ddad/ddad_surround_fusion.yaml \\
        --output ./debug/ddad_val_batch.pt \\
        --split val --batch_index 0 --batch_size 1
"""
import argparse
import os
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

import utils
from dataset import construct_dataset


def _build_dataset(cfg: Dict[str, Any], split: str, apply_aug: bool, use_synth: bool):
    """Build dataset with the same transforms used during training/eval."""
    height = int(cfg['training']['height'])
    width = int(cfg['training']['width'])

    if split == 'train' and apply_aug:
        jitter = (0.2, 0.2, 0.2, 0.05)
    else:
        jitter = (0.0, 0.0, 0.0, 0.0)

    augmentation = {
        'image_shape': (height, width),
        'jittering': jitter,
        'crop_train_borders': (),
        'crop_eval_borders': ()
    }

    if split not in ('train', 'val'):
        raise ValueError(f'Unsupported split "{split}", expected "train" or "val".')

    synth_mode = use_synth or cfg['data']['dataset'] == 'synthetic'
    if synth_mode:
        from dataset.synth_dataset import SyntheticSurroundDataset
        num_samples = cfg['data'].get(
            'synthetic_length', 16 if split == 'train' else 8)
        synth_kwargs = augmentation.copy()
        synth_kwargs.pop('image_shape', None)
        return SyntheticSurroundDataset(
            cfg,
            split,
            image_shape=(height, width),
            num_samples=num_samples,
            **synth_kwargs,
        )

    return construct_dataset(cfg, split, **augmentation)


def _move_to_cpu(sample: Any):
    """Detach tensors and move them to CPU so torch.save can serialize them."""
    if torch.is_tensor(sample):
        return sample.detach().cpu()
    if isinstance(sample, dict):
        return {k: _move_to_cpu(v) for k, v in sample.items()}
    if isinstance(sample, list):
        return [_move_to_cpu(v) for v in sample]
    if isinstance(sample, tuple):
        return tuple(_move_to_cpu(v) for v in sample)
    return sample


def _summarize(sample: Dict[str, Any]):
    """Print a short summary of the saved batch for quick inspection."""
    print('Saved keys and shapes:')
    for key, val in sample.items():
        if torch.is_tensor(val):
            print(f'  {key}: tensor{tuple(val.shape)} dtype={val.dtype}')
        elif isinstance(val, list) and val and torch.is_tensor(val[0]):
            shapes = [tuple(v.shape) for v in val]
            print(f'  {key}: list[{len(val)}] of tensors {shapes}')
        else:
            print(f'  {key}: type={type(val)}')


def parse_args():
    parser = argparse.ArgumentParser(description='Dump a VFDepth batch for debugging.')
    parser.add_argument('--config_file', type=str,
                        default='./configs/ddad/ddad_surround_fusion.yaml',
                        help='Path to the VFDepth yaml config.')
    parser.add_argument('--output', type=str, required=True,
                        help='Where to store the serialized batch (torch.save).')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'],
                        help='Dataset split to sample from.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of samples to pull from the DataLoader.')
    parser.add_argument('--batch_index', type=int, default=0,
                        help='0-based batch index to grab from the loader.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for the DataLoader.')
    parser.add_argument('--apply_aug', action='store_true',
                        help='Enable photometric augmentation when split=train.')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use the built-in synthetic dataset instead of real DDAD/NuScenes data.')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = utils.get_config(args.config_file, mode='train')

    dataset = _build_dataset(cfg, args.split, args.apply_aug, args.synthetic)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False
    )

    if args.batch_index < 0:
        raise ValueError('batch_index must be >= 0')

    print(f'Collecting batch #{args.batch_index} from {args.split} split '
          f'({len(dataset)} samples)...')
    batch = None
    for idx, sample in enumerate(loader):
        if idx == args.batch_index:
            batch = sample
            break

    if batch is None:
        raise RuntimeError('Requested batch_index is larger than the loader length.')

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    cpu_batch = _move_to_cpu(batch)
    torch.save(cpu_batch, args.output)
    print(f'Saved batch to {args.output}')
    _summarize(cpu_batch)


if __name__ == '__main__':
    main()
