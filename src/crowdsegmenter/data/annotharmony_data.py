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
from torchvision.transforms import InterpolationMode
from typing import Union, List, Tuple, Optional, Dict
from crowdsegmenter.config import DataConfig 

class AnnotHarmonyDataset(Dataset):
    """Dataset for handling multi-annotator segmentation data.

    This dataset loads images and their corresponding segmentation masks from multiple
    annotators and ground truth, supporting synchronization of transforms.
    """
    
    def __init__(
        self, 
        config: DataConfig, 
        partition: str, 
        transform: transforms.Compose | None = None
    ):
        self.config = config
        self.partition = partition
        self.images_folder = config.images_folder
        self.masks_folder = config.masks_folder
        self.ground_truth_folder = config.ground_truth_folder
        self.normalize = config.normalize
        self.transform = transform
        self.data_path = Path(config.data_dir) / self.partition

        if not self.data_path.exists():
            raise FileNotFoundError(f"Partition directory not found: {self.data_path}")

        supported_formats = ['*.png', '*.jpg', '*.jpeg']
        self.patch_files = []
        for fmt in supported_formats:
            self.patch_files.extend(glob.glob(str(self.data_path / self.images_folder / fmt)))
        
        self.patch_files = sorted(self.patch_files, key=self._alphanumeric_key)
        self.file_names = [Path(f).name for f in self.patch_files]

        self.masks_path, self.gt_masks_path = self._prepare_mask_paths()

    def _alphanumeric_key(self, s: str) -> List[Union[int, str]]:
        """Splits a string into a list of integers and strings for natural sorting.

        Args:
            s: The string to be split.

        Returns:
            A list containing integers and strings.
        """
        return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', s)]

    def _prepare_mask_paths(self) -> Tuple[List[List[str]], List[List[Path]]]:
        """Indexes all mask and ground truth paths for the dataset partition.

        Returns:
            A tuple containing:
                - masks_path: List of lists containing paths to annotator masks.
                - gt_masks_path: List of lists containing paths to ground truth masks.
        """
        mask_root = self.data_path / self.masks_folder
        list_annotators = sorted([
            ann for ann in os.listdir(mask_root)
            if os.path.isdir(mask_root / ann) and ann != self.ground_truth_folder
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
                for c in range(self.config.num_classes):
                    p = mask_root / self.ground_truth_folder / f'class_{c}' / sample
                    sample_gt.append(p)
                gt_masks_path.append(sample_gt)

        return masks_path, gt_masks_path

    def _load_tensor(self, path: str, mode: ImageReadMode, normalize: bool = True) -> torch.Tensor:
        """
        Loads an image or mask from a file and converts it to a torch Tensor.

        Args:
            path: Path to the image file.
            mode: Image read mode (e.g., RGB or GRAY).
            normalize: Whether to normalize pixel values to [0, 1]. Defaults to True.

        Returns:
            A torch Tensor representing the image.
        """
        if not Path(path).exists():
            if mode == ImageReadMode.GRAY:
                return torch.full((1, *self.config.image_size), self.config.ignored_value, dtype=torch.float32)
            return torch.zeros((3, *self.config.image_size), dtype=torch.float32)
        
        tensor = torchvision.io.read_image(path, mode=mode)
        interpolation = InterpolationMode.NEAREST if mode == ImageReadMode.GRAY else InterpolationMode.BILINEAR
        tensor = TF.resize(tensor, list(self.config.image_size), interpolation = interpolation)
        tensor = tensor.float() / 255.0 if normalize else tensor.float()
        return tensor

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset.

        Returns:
            The number of patch files.
        """
        return len(self.patch_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Retrieves a single data sample (image, masks, and ground truth) by index.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A tuple containing the image tensor, annotator masks tensor (if enabled),
            annotator existence one-hot tensor (if enabled), and ground truth tensor (if enabled).
        """

        image = self._load_tensor(self.patch_files[idx], ImageReadMode.RGB, normalize = self.normalize)
        
        results = [image]


        if self.config.load_annotators:
            m_paths = self.masks_path[idx]
            masks = torch.zeros(self.config.num_annotators * self.config.num_classes, *self.config.image_size)
            anns_onehot = torch.zeros(self.config.num_annotators)
            
            for i, p in enumerate(m_paths):
                mask = self._load_tensor(p, ImageReadMode.GRAY, normalize = self.normalize)
                masks[i] = mask.squeeze(0)
                
                ann_idx = i % self.config.num_annotators
                if torch.any(mask == self.config.ignored_value):
                    anns_onehot[ann_idx] = 0.0
                else:
                    anns_onehot[ann_idx] = 1.0
            
            results.extend([masks, anns_onehot])

        if self.config.load_ground_truth:
            gt_paths = self.gt_masks_path[idx]
            gt_tensor = torch.zeros(len(gt_paths), *self.config.image_size)
            for i, p in enumerate(gt_paths):
                gt_tensor[i] = self._load_tensor(str(p), ImageReadMode.GRAY, normalize = self.normalize).squeeze(0)
            results.append(gt_tensor)

        if self.transform and self.partition in ['Train', 'train', 'Training', 'training']:
            results = self._apply_sync_transform(results)

        return tuple(results)

    def _apply_sync_transform(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Applies random horizontal and vertical flips synchronously to all input tensors.

        Args:
            tensors: A list of tensors (image, masks, ground truth) to be transformed.

        Returns:
            The list of transformed tensors.
        """

        if random.random() > 0.5:
            tensors = [TF.hflip(t) if t.ndim > 1 else t for t in tensors]
        if random.random() > 0.5:
            tensors = [TF.vflip(t) if t.ndim > 1 else t for t in tensors]
        
        tensors[0] = self.transform(tensors[0])
        return tensors


class AnnotHarmonyDataLoader:
    """Utility class to manage DataLoaders for different dataset partitions.

    This class handles the creation of training, validation, and testing split loaders
    with appropriate transforms and configurations.
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.transforms = config.transforms
        self.partitions = config.partitions

    def _get_transforms(self) -> Dict[str, transforms.Compose]:
        """Defines the data transformations for different pipeline stages.

        Returns:
            A dictionary mapping partition keys ('train', 'inference') to their transforms.
        """

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
        """Creates DataLoaders for all defined dataset partitions.

        Returns:
            A tuple containing (train_loader, val_loader, test_loader).
        """

        loaders = {}

        for split in self.partitions:
            is_train = split in ['Train', 'train', 'Training', 'training']
            transform = None
            if self.transforms:
                transform_type = 'train' if is_train else 'inference'
                transform = self._get_transforms()[transform_type]

            tag = split.lower()
            if 'train' in tag: key = 'train'
            elif 'val' in tag: key = 'val'
            elif 'test' in tag: key = 'test'
            else: key = tag
            
            dataset = AnnotHarmonyDataset(
                config=self.config,
                partition=split,
                transform=transform
            )

            loaders[key] = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=is_train,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else None,
                persistent_workers=True if self.config.num_workers > 0 else False
            )

        return loaders['train'], loaders['val'], loaders['test']