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

class CrowdSegDataset(Dataset):
    """Dataset for stochastic training of annotator reliability models.
    
    Instead of returning all masks, it randomly selects ONE valid annotator 
    per sample and returns their specific mask and One-Hot ID.
    """
    
    def __init__(
        self, 
        config: DataConfig, 
        partition: str, 
        images_folder: str,
        masks_folder: str,
        ground_truth_folder: str,
        normalize: bool = True,
        transform: transforms.Compose | None = None
    ):
        self.config = config
        self.partition = partition
        self.images_folder = images_folder
        self.masks_folder = masks_folder
        self.ground_truth_folder = ground_truth_folder
        self.normalize = normalize
        self.transform = transform
        self.data_path = Path(config.data_dir) / self.partition

        if not self.data_path.exists():
            raise FileNotFoundError(f"Partition directory not found: {self.data_path}")

        # Collect image files
        supported_formats = ['*.png', '*.jpg', '*.jpeg']
        self.patch_files = []
        for fmt in supported_formats:
            self.patch_files.extend(glob.glob(str(self.data_path / self.images_folder / fmt)))
        
        self.patch_files = sorted(self.patch_files, key=self._alphanumeric_key)
        self.file_names = [Path(f).name for f in self.patch_files]

        # Index available annotators per sample
        self.masks_db, self.gt_masks_path = self._index_annotators()

    def _alphanumeric_key(self, s: str) -> List[Union[int, str]]:
        """Splits a string into a list of integers and strings for natural sorting.

        Args:
            s: The string to be split.

        Returns:
            A list containing integers and strings.
        """
        return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', s)]

    def _index_annotators(self) -> Tuple[List[Dict[int, List[str]]], List[List[Path]]]:
        """Indexes which annotators labeled each image to allow fast random selection.

        Returns:
            A tuple containing:
                - masks_db: List of dictionaries mapping annotator IDs to mask paths.
                - gt_masks_path: List of lists containing paths to ground truth masks.
        """
        mask_root = self.data_path / self.masks_folder
        
        # Get sorted list of annotators to maintain consistent One-Hot indexing
        list_annotators = sorted([
            ann for ann in os.listdir(mask_root)
            if os.path.isdir(mask_root / ann) and ann != self.ground_truth_folder
        ], key=self._alphanumeric_key)

        masks_db = []
        gt_masks_path = []

        for sample in tqdm(self.file_names, desc=f"Indexing {self.partition} annotators"):
            # Map valid annotator IDs to their specific class mask paths
            available_for_sample = {}
            for idx, ann_name in enumerate(list_annotators):
                # We check class_0 existence as a signal of annotation availability
                proxy_path = mask_root / ann_name / 'class_0' / sample
                if proxy_path.exists():
                    available_for_sample[idx] = [
                        str(mask_root / ann_name / f'class_{c}' / sample) 
                        for c in range(self.config.num_classes)
                    ]
            masks_db.append(available_for_sample)

            # Standard Ground Truth indexing
            if self.config.load_ground_truth:
                sample_gt = []
                for c in range(self.config.num_classes):
                    p = mask_root / self.ground_truth_folder / f'class_{c}' / sample
                    sample_gt.append(p)
                gt_masks_path.append(sample_gt)

        return masks_db, gt_masks_path

    def _load_tensor(self, path: str, mode: ImageReadMode, normalize: bool = True) -> torch.Tensor:
        """Loads an image or mask from a file and converts it to a torch Tensor.

        Args:
            path: Path to the image file.
            mode: Image read mode (e.g., RGB or GRAY).
            normalize: Whether to normalize pixel values to [0, 1]. Defaults to True.

        Returns:
            A torch Tensor representing the image.
        """
        if not Path(path).exists():
            val = self.config.ignored_value if mode == ImageReadMode.GRAY else 0.0
            channels = 1 if mode == ImageReadMode.GRAY else 3
            return torch.full((channels, *self.config.image_size), val, dtype=torch.float32)
        
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
        """Retrieves a single data sample (image, one annotator mask, and ground truth) by index.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A tuple containing the image tensor, a randomly selected annotator mask tensor,
            the one-hot ID of the selected annotator, and the ground truth tensor (if enabled).
        """
        # Load Base Image
        image = self._load_tensor(self.patch_files[idx], ImageReadMode.RGB, normalize = self.normalize)
        results = [image]

        # One-Hot Annotator Logic
        if self.config.load_annotators:
            available = self.masks_db[idx]
            
            # Select ONE annotator randomly from available ones for this specific sample
            # If no annotators available (rare), fallback to index 0 with ignored values
            if available:
                chosen_idx = random.choice(list(available.keys()))
                ann_paths = available[chosen_idx]
                ann_mask = torch.zeros(self.config.num_classes, *self.config.image_size)
                for i, p in enumerate(ann_paths):
                    ann_mask[i] = self._load_tensor(p, ImageReadMode.GRAY, normalize = self.normalize).squeeze(0)
            else:
                chosen_idx = 0
                ann_mask = torch.full((self.config.num_classes, *self.config.image_size), self.config.ignored_value)

            # Create One-Hot ID vector [num_annotators]
            one_hot = torch.zeros(self.config.num_annotators)
            one_hot[chosen_idx] = 1.0
            results.extend([ann_mask, one_hot])

        # Load Ground Truth
        if self.config.load_ground_truth:
            gt_paths = self.gt_masks_path[idx]
            gt_tensor = torch.zeros(len(gt_paths), *self.config.image_size)
            for i, p in enumerate(gt_paths):
                gt_tensor[i] = self._load_tensor(str(p), ImageReadMode.GRAY, normalize = self.normalize).squeeze(0)
            results.append(gt_tensor)

        # Synchronized Geometric Augmentation
        if self.transform and self.partition == 'train':
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


class CrowdSegDataLoader:
    """Manager for CrowdSeg DataLoaders with robust partition mapping.

    This class handles the creation of training, validation, and testing split loaders
    using the CrowdSegDataset.
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.transforms = self._get_transforms()
        self.partitions = config.partitions
        self.images_folder = config.images_folder
        self.masks_folder = config.masks_folder 
        self.ground_truth_folder = config.ground_truth_folder
        self.normalize = config.normalize

    def _get_transforms(self) -> Dict[str, transforms.Compose]:
        """Defines the data transformations for different pipeline stages.

        Returns:
            A dictionary mapping partition keys ('train', 'inference') to their transforms.
        """
        return {
            'train': transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.Normalize(self.config.mean, self.config.std)
            ]),
            'inference': transforms.Compose([
                transforms.Normalize(self.config.mean, self.config.std)
            ]),
        }

    def get_split_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Creates DataLoaders for all defined dataset partitions.

        Returns:
            A tuple containing (train_loader, val_loader, test_loader).
        """
        loaders = {}

        for split in self.partitions:
            is_train = split.lower() in ['train', 'training']
            transform_type = 'train' if is_train else 'inference'
            
            # Robust key mapping
            tag = split.lower()
            if 'train' in tag: key = 'train'
            elif 'val' in tag: key = 'val'
            elif 'test' in tag: key = 'test'
            else: key = tag

            dataset = CrowdSegDataset(
                config=self.config,
                partition=split,
                images_folder=self.images_folder,
                masks_folder=self.masks_folder,
                ground_truth_folder=self.ground_truth_folder,
                normalize = self.normalize,
                transform=self.transforms[transform_type]
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