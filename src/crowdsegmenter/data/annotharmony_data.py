import os
import re
import glob
import torch
import random
import torchvision.io
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.io import ImageReadMode
from typing import Union, List, Tuple, Optional, Dict
from crowdsegmenter.config import DataConfig 


class AnnotHarmonyDataset(Dataset):
    
    def __init__(
        self, 
        config: DataConfig, 
        partition: str, 
        transform: transforms.Compose | None = None
    ):
        self.config = config
        self.partition = partition
        self.transform = transform
        self.data_path = Path(config.data_dir) / partition
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Partition directory not found: {self.data_path}")

        supported_formats = ['*.png', '*.jpg', '*.jpeg']
        self.patch_files = []
        for fmt in supported_formats:
            self.patch_files.extend(glob.glob(str(self.data_path / config.images_folder / fmt)))
        
        self.patch_files = sorted(self.patch_files, key=self._alphanumeric_key)
        self.file_names = [Path(f).name for f in self.patch_files]

        self.masks_path, self.gt_masks_path = self._prepare_mask_paths()

    def _alphanumeric_key(self, s: str) -> List[Union[int, str]]:
        return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', s)]

    def _prepare_mask_paths(self) -> Tuple[List[List[str]], List[List[Path]]]:
        mask_root = self.data_path / 'masks'
        list_annotators = sorted([
            ann for ann in os.listdir(mask_root)
            if os.path.isdir(mask_root / ann) and ann.lower() != 'ground_truth'
        ], key=self._alphanumeric_key)

        masks_path = []
        gt_masks_path = []

        for sample in tqdm(self.file_names, desc=f"Indexing {self.partition} masks"):
            if self.config.load_annotators:
                sample_masks = []
                for c in range(self.config.num_classes):
                    for ann in list_annotators:
                        p = mask_root / ann / f'class_{c}' / sample
                        sample_masks.append(str(p))
                masks_path.append(sample_masks)

            if self.config.load_ground_truth:
                sample_gt = []
                class_ids = [self.config.single_class] if self.config.single_class is not None else range(self.config.num_classes)
                for c in class_ids:
                    p = mask_root / 'ground_truth' / f'class_{c}' / sample
                    sample_gt.append(p)
                gt_masks_path.append(sample_gt)

        return masks_path, gt_masks_path

    def _load_tensor(self, path: str, mode: ImageReadMode, normalize: bool = True) -> torch.Tensor:
        if not Path(path).exists():
            if mode == ImageReadMode.GRAY:
                return torch.full((1, *self.config.image_size), self.config.ignored_value, dtype=torch.float32)
            return torch.zeros((3, *self.config.image_size), dtype=torch.float32)
        
        tensor = torchvision.io.read_image(path, mode=mode)
        tensor = TF.resize(tensor, list(self.config.image_size))
        return tensor.float() / 255.0 if normalize else tensor.float()

    def __len__(self) -> int:
        return len(self.patch_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        image = self._load_tensor(self.patch_files[idx], ImageReadMode.RGB)
        
        results = [image]

        if self.config.load_annotators:
            m_paths = self.masks_path[idx]
            masks = torch.zeros(self.config.num_annotators * self.config.num_classes, *self.config.image_size)
            anns_onehot = torch.zeros(self.config.num_annotators)
            
            for i, p in enumerate(m_paths):
                mask = self._load_tensor(p, ImageReadMode.GRAY)
                masks[i] = mask.squeeze(0)
                
                ann_idx = i % self.config.num_annotators
                if torch.all(mask != self.config.ignored_value):
                    anns_onehot[ann_idx] = 1.0
            
            results.extend([masks, anns_onehot])

        if self.config.load_ground_truth:
            gt_paths = self.gt_masks_path[idx]
            gt_tensor = torch.zeros(len(gt_paths), *self.config.image_size)
            for i, p in enumerate(gt_paths):
                gt_tensor[i] = self._load_tensor(str(p), ImageReadMode.GRAY).squeeze(0)
            results.append(gt_tensor)

        if self.transform and self.partition == 'train':
            results = self._apply_sync_transform(results)

        return tuple(results)

    def _apply_sync_transform(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        if random.random() > 0.5:
            tensors = [TF.hflip(t) if t.ndim > 1 else t for t in tensors]
        if random.random() > 0.5:
            tensors = [TF.vflip(t) if t.ndim > 1 else t for t in tensors]
        
        tensors[0] = self.transform(tensors[0])
        return tensors


class AnnotHarmonyDataLoader:
    def __init__(self, config: DataConfig):
        self.config = config
        self.transforms = self._get_transforms()

    def _get_transforms(self) -> Dict[str, transforms.Compose]:
        return {
            'train': transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.Normalize(self.config.mean, self.config.std) if hasattr(self.config, 'mean') else transforms.Lambda(lambda x: x)
            ]),
            'inference': transforms.Compose([
                transforms.Normalize(self.config.mean, self.config.std) if hasattr(self.config, 'mean') else transforms.Lambda(lambda x: x)
            ]),
        }

    def get_split_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        splits = ['train', 'val', 'test']
        loaders = {}

        for split in splits:
            is_train = split == 'train'
            transform_type = 'train' if is_train else 'inference'
            
            dataset = AnnotHarmonyDataset(
                config=self.config,
                partition=split,
                transform=self.transforms[transform_type]
            )

            loaders[split] = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=is_train,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else None,
                persistent_workers=True if self.config.num_workers > 0 else False
            )

        return loaders['train'], loaders['val'], loaders['test']