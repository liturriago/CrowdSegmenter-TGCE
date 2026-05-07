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
from typing import Union, List, Tuple, Optional, Dict, Literal
from crowdsegmenter.config import DataConfig


class CrowdSegmenterDataset(Dataset):
    """Unified multi-annotator segmentation dataset for AnnotHarmony and CrowdSeg.

    Supports two operating modes controlled by the ``mode`` argument:

    - **``"Annot-Harmony"``** — loads all annotators simultaneously, producing
      a mask of shape ``[A*C, H, W]`` ordered as
      ``(ann_0_class_0, ann_1_class_0, ..., ann_R_class_0, ann_0_class_1, ...)``.
      A multi-hot vector ``[A]`` marks which annotators provided valid labels.
      This is the format expected by ``TGCE_SSPS``.

    - **``"CrowdSeg"``** — randomly selects one valid annotator per sample,
      producing a mask of shape ``[A*C, H, W]`` where only the selected
      annotator's channels carry real values and the rest are filled with
      ``ignored_value``. A one-hot vector ``[A]`` identifies the selection.
      ``NoisyLabelLoss`` uses this vector to extract the relevant channels.

    Both modes share the same output contract:
        ``(image, masks, anns_ids [, ground_truth])``

    where ``masks`` is always ``[A*C, H, W]``, making the downstream trainer
    and metric tracker identical for both architectures.

    Args:
        config (DataConfig): Dataset configuration.
        partition (str): Dataset split name (e.g. ``"Train"``, ``"Valid"``).
        mode (Literal["Annot-Harmony", "CrowdSeg"]): Operating mode.
        transform (transforms.Compose | None): Photometric transform applied
            to the image only (geometric augmentations are applied in sync
            to all tensors separately).
    """

    MODES = ("Annot-Harmony", "CrowdSeg")

    def __init__(
        self,
        config: DataConfig,
        partition: str,
        mode: Literal["Annot-Harmony", "CrowdSeg"] = "Annot-Harmony",
        transform: transforms.Compose | None = None,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Choose from {self.MODES}."
            )

        self.config     = config
        self.partition  = partition
        self.mode       = mode
        self.transform  = transform
        self.normalize  = config.normalize
        self.data_path  = Path(config.data_dir) / partition

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Partition directory not found: {self.data_path}"
            )

        # Collect and sort image files
        self.patch_files: List[str] = []
        for fmt in ("*.png", "*.jpg", "*.jpeg"):
            self.patch_files.extend(
                glob.glob(str(self.data_path / config.images_folder / fmt))
            )
        self.patch_files = sorted(self.patch_files, key=self._alphanumeric_key)
        self.file_names  = [Path(f).name for f in self.patch_files]

        # Index annotators and ground truth paths
        self.annotators,       \
        self.masks_path,       \
        self.gt_masks_path = self._index_masks()

    # ------------------------------------------------------------------ #
    #  Indexing                                                            #
    # ------------------------------------------------------------------ #

    def _alphanumeric_key(self, s: str) -> List[Union[int, str]]:
        """Natural-sort key that handles embedded integers correctly."""
        return [
            int(p) if p.isdigit() else p
            for p in re.split(r"(\d+)", s)
        ]

    def _index_masks(
        self,
    ) -> Tuple[List[str], List[List[str]], List[List[Path]]]:
        """Indexes annotator directories and builds per-sample path tables.

        Returns:
            Tuple of:
                - annotators: Sorted list of annotator directory names.
                - masks_path: For each sample, ordered list of mask paths
                  following the layout ``(ann_0_class_0, ann_1_class_0, ...
                  ann_0_class_K, ..., ann_R_class_K)``.
                - gt_masks_path: For each sample, list of ground truth paths
                  per class (empty list if ``load_ground_truth`` is False).
        """
        mask_root = self.data_path / self.config.masks_folder

        annotators = sorted(
            [
                ann for ann in os.listdir(mask_root)
                if os.path.isdir(mask_root / ann)
                and ann != self.config.ground_truth_folder
            ],
            key=self._alphanumeric_key,
        )

        masks_path:    List[List[str]]  = []
        gt_masks_path: List[List[Path]] = []

        for sample in tqdm(
            self.file_names, desc=f"Indexing {self.partition} masks"
        ):
            # Annotator masks — layout: class_0 × all_anns, class_1 × all_anns …
            sample_masks: List[str] = []
            for c in range(self.config.num_classes):
                for ann in annotators:
                    sample_masks.append(
                        str(mask_root / ann / f"class_{c}" / sample)
                    )
            masks_path.append(sample_masks)

            # Ground truth (optional)
            if self.config.load_ground_truth:
                gt_masks_path.append([
                    mask_root / self.config.ground_truth_folder / f"class_{c}" / sample
                    for c in range(self.config.num_classes)
                ])
            else:
                gt_masks_path.append([])

        return annotators, masks_path, gt_masks_path

    # ------------------------------------------------------------------ #
    #  I/O                                                                 #
    # ------------------------------------------------------------------ #

    def _load_tensor(
        self,
        path: str,
        mode: ImageReadMode,
    ) -> torch.Tensor:
        """Loads an image or mask and returns a float tensor.

        Missing files return a sentinel-filled tensor for masks and a
        zero tensor for RGB images, so the pipeline never crashes on
        incomplete annotation sets.

        Args:
            path: Absolute path to the file.
            mode: ``ImageReadMode.RGB`` or ``ImageReadMode.GRAY``.

        Returns:
            Float tensor of shape ``[C, H, W]`` in ``[0, 1]`` if
            ``normalize`` is True, otherwise raw float values.
        """
        if not Path(path).exists():
            if mode == ImageReadMode.GRAY:
                return torch.full(
                    (1, *self.config.image_size),
                    self.config.ignored_value,
                    dtype=torch.float32,
                )
            return torch.zeros(
                (3, *self.config.image_size), dtype=torch.float32
            )

        tensor = torchvision.io.read_image(path, mode=mode)
        interp = (
            InterpolationMode.NEAREST
            if mode == ImageReadMode.GRAY
            else InterpolationMode.BILINEAR
        )
        tensor = TF.resize(tensor, list(self.config.image_size), interpolation=interp)
        return tensor.float() / 255.0 if self.normalize else tensor.float()

    # ------------------------------------------------------------------ #
    #  Dataset interface                                                   #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.patch_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Returns a single sample.

        Output contract (shared by both modes):
            - ``batch[0]`` image       ``[C_in, H, W]``
            - ``batch[1]`` masks       ``[A*C,  H, W]``
            - ``batch[2]`` anns_ids    ``[A]``
            - ``batch[3]`` ground_truth ``[C, H, W]``  (only if load_ground_truth)

        In **Annot-Harmony** mode all annotator channels carry real values
        (or ``ignored_value`` where the annotation is missing). ``anns_ids``
        is a multi-hot vector marking annotators that provided at least one
        valid pixel.

        In **CrowdSeg** mode only the channels belonging to the selected
        annotator carry real values; all other channels are filled with
        ``ignored_value``. ``anns_ids`` is a one-hot vector.
        """
        image = self._load_tensor(self.patch_files[idx], ImageReadMode.RGB)
        results: List[torch.Tensor] = [image]

        if self.config.load_annotators:
            masks, anns_ids = (
                self._load_all_annotators(idx)
                if self.mode == "Annot-Harmony"
                else self._load_one_annotator(idx)
            )
            results.extend([masks, anns_ids])

        if self.config.load_ground_truth:
            gt = torch.stack([
                self._load_tensor(str(p), ImageReadMode.GRAY).squeeze(0)
                for p in self.gt_masks_path[idx]
            ], dim=0)
            results.append(gt)

        if self.transform and self.partition.lower() in (
            "train", "training"
        ):
            results = self._apply_sync_transform(results)

        return tuple(results)

    # ------------------------------------------------------------------ #
    #  Mode-specific loading                                               #
    # ------------------------------------------------------------------ #

    def _load_all_annotators(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """AnnotHarmony mode: loads every annotator's masks.

        Returns:
            masks    ``[A*C, H, W]`` — all annotators, sentinel where missing.
            anns_ids ``[A]``         — multi-hot (1 = annotator has valid labels).
        """
        A, C = self.config.num_annotators, self.config.num_classes
        H, W = self.config.image_size

        masks    = torch.full((A * C, H, W), self.config.ignored_value)
        anns_ids = torch.zeros(A)

        for ch_idx, path in enumerate(self.masks_path[idx]):
            mask = self._load_tensor(path, ImageReadMode.GRAY).squeeze(0)
            masks[ch_idx] = mask

            # Annotator index from channel position: ch = k*A + r → r = ch % A
            ann_r = ch_idx % A
            if not torch.any(mask == self.config.ignored_value):
                anns_ids[ann_r] = 1.0

        return masks, anns_ids

    def _load_one_annotator(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """CrowdSeg mode: loads one randomly selected valid annotator.

        The selected annotator's masks are placed in the correct channels
        of a sentinel-filled ``[A*C, H, W]`` tensor, so the layout is
        identical to AnnotHarmony mode and ``NoisyLabelLoss`` can locate
        them via the one-hot vector.

        Returns:
            masks    ``[A*C, H, W]`` — sentinel everywhere except the
                     selected annotator's channels.
            anns_ids ``[A]``         — one-hot.
        """
        A, C = self.config.num_annotators, self.config.num_classes
        H, W = self.config.image_size

        masks    = torch.full((A * C, H, W), self.config.ignored_value)
        anns_ids = torch.zeros(A)

        # Identify annotators that have at least class_0 available
        available = [
            r for r, ann in enumerate(self.annotators)
            if Path(self.masks_path[idx][r]).exists()   # channel 0*A + r = r
        ]

        chosen_r = random.choice(available) if available else 0

        # Fill only the chosen annotator's channels: channel = k*A + chosen_r
        for k in range(C):
            ch_idx = k * A + chosen_r
            path   = self.masks_path[idx][ch_idx]
            masks[ch_idx] = self._load_tensor(path, ImageReadMode.GRAY).squeeze(0)

        anns_ids[chosen_r] = 1.0

        return masks, anns_ids

    # ------------------------------------------------------------------ #
    #  Augmentation                                                        #
    # ------------------------------------------------------------------ #

    def _apply_sync_transform(
        self, tensors: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Applies synchronised random flips to all spatial tensors.

        The photometric transform (``self.transform``) is applied only to
        the image (``tensors[0]``); masks and ground truth receive only the
        geometric flips.

        Args:
            tensors: ``[image, masks, anns_ids, ...]`` — ``anns_ids`` is
                1-D so the flip guards skip it automatically.

        Returns:
            Augmented tensor list.
        """
        if random.random() > 0.5:
            tensors = [TF.hflip(t) if t.ndim > 1 else t for t in tensors]
        if random.random() > 0.5:
            tensors = [TF.vflip(t) if t.ndim > 1 else t for t in tensors]

        tensors[0] = self.transform(tensors[0])
        return tensors


class CrowdSegmenterDataLoader:
    """DataLoader manager for the unified CrowdSegmenterDataset.

    Creates train / val / test loaders for either operating mode, applying
    appropriate photometric transforms per split.

    Args:
        config (DataConfig): Dataset configuration.
        mode (Literal["Annot-Harmony", "CrowdSeg"]): Dataset operating mode
            forwarded to every ``CrowdSegmenterDataset`` instance.
    """

    def __init__(
        self,
        config: DataConfig,
        mode: Literal["Annot-Harmony", "CrowdSeg"] = "Annot-Harmony",
    ) -> None:
        self.config     = config
        self.mode       = mode
        self.transforms = config.transforms
        self.partitions = config.partitions

    def _get_transforms(self) -> Dict[str, transforms.Compose]:
        train_tfms = [
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
        infer_tfms: list = []

        if self.config.mean and self.config.std:
            norm = transforms.Normalize(self.config.mean, self.config.std)
            train_tfms.append(norm)
            infer_tfms.append(norm)

        return {
            "train":     transforms.Compose(train_tfms),
            "inference": transforms.Compose(infer_tfms)
            if infer_tfms
            else transforms.Compose([transforms.Lambda(lambda x: x)]),
        }

    def get_split_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Builds and returns ``(train_loader, val_loader, test_loader)``.

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]:
                One loader per split in the order train / val / test.
        """
        loaders: Dict[str, DataLoader] = {}
        tfms = self._get_transforms() if self.transforms else None

        for split in self.partitions:
            is_train = split.lower() in ("train", "training")

            transform = None
            if tfms is not None:
                transform = tfms["train"] if is_train else tfms["inference"]

            tag = split.lower()
            if "train" in tag:   key = "train"
            elif "val" in tag:   key = "val"
            elif "test" in tag:  key = "test"
            else:                key = tag

            dataset = CrowdSegmenterDataset(
                config=self.config,
                partition=split,
                mode=self.mode,
                transform=transform,
            )

            loaders[key] = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=is_train,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                prefetch_factor=(
                    self.config.prefetch_factor
                    if self.config.num_workers > 0 else None
                ),
                persistent_workers=self.config.num_workers > 0,
            )

        return loaders["train"], loaders["val"], loaders["test"]