import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional, Set, Union
from crowdsegmenter.config import TrainConfig

def get_training_phase(
    model: nn.Module,
    config: TrainConfig,
    phase: Optional[int] = None,
) -> Tuple[optim.Optimizer, Optional[optim.lr_scheduler.LRScheduler]]:
    """Applies gradual unfreezing to CrowdSeg/AnnotHarmony based on the training phase.

    Implements a curriculum learning strategy by progressively unfreezing
    model components from the top (segmentation/annotator heads) down to
    the bottom (encoder backbone), assigning differential learning rates
    to each group.

    If ``phase`` is ``None``, the entire model is trained end-to-end with a
    single optimizer and scheduler — no curriculum is applied.

    The unfreezing curriculum follows this order:
        - Phase 1: Heads only (segmentation + annotator).
        - Phase 2: Heads + decoder.
        - Phase 3: Heads + decoder + encoder normalization layers.
        - Phase 4: Heads + decoder + encoder last block + norms. Adds scheduler.
        - Phase 5: Full model (all encoder blocks). Adds scheduler.

    Args:
        model (nn.Module): A CrowdSeg or AnnotHarmony instance. Must expose
            ``encoder``, ``decoder``, ``segmentation_head``, and
            ``annotator_head`` attributes.
        config (TrainConfig): Configuration object containing ``lr``,
            ``transfer_lr``, and ``gamma`` for the scheduler.
        phase (Optional[int]): Training phase index (1–5). Pass ``None``
            to skip curriculum and train the full model directly.

    Returns:
        Tuple[optim.Optimizer, Optional[LRScheduler]]:
            Configured Adam optimizer and an optional ExponentialLR scheduler
            (only attached in phases 4, 5, and the phase-free mode).
    """
    scheduler = None

    # ------------------------------------------------------------------ #
    #  Phase-free mode: full model, single LR, with scheduler             #
    # ------------------------------------------------------------------ #
    if phase is None:
        for param in model.parameters():
            param.requires_grad = True

        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)
        return optimizer, scheduler

    # ------------------------------------------------------------------ #
    #  Shared helpers                                                      #
    # ------------------------------------------------------------------ #
    def _freeze_all() -> None:
        for param in model.parameters():
            param.requires_grad = False

    def _unfreeze(module: nn.Module) -> list:
        params = list(module.parameters())
        for param in params:
            param.requires_grad = True
        return params

    def _unfreeze_norms(exclude: Set[nn.Parameter]) -> list:
        """Unfreezes BatchNorm/LayerNorm params in the encoder not already tracked."""
        norm_params = []
        for m in model.encoder.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                for param in m.parameters():
                    if param not in exclude:
                        param.requires_grad = True
                        norm_params.append(param)
        return norm_params

    # ------------------------------------------------------------------ #
    #  Curriculum phases                                                   #
    # ------------------------------------------------------------------ #
    match phase:

        case 1:
            # Phase 1: Train heads only. Everything else frozen.
            _freeze_all()
            head_params = _unfreeze(model.segmentation_head) + _unfreeze(model.annotator_head)

            optimizer = optim.Adam([
                {'params': head_params, 'lr': config.lr},
            ])

        case 2:
            # Phase 2: Heads + decoder.
            _freeze_all()
            head_params   = _unfreeze(model.segmentation_head) + _unfreeze(model.annotator_head)
            decoder_params = _unfreeze(model.decoder)

            optimizer = optim.Adam([
                {'params': head_params,    'lr': config.lr},
                {'params': decoder_params, 'lr': config.transfer_lr},
            ])

        case 3:
            # Phase 3: Heads + decoder + encoder normalization layers.
            _freeze_all()
            head_params    = _unfreeze(model.segmentation_head) + _unfreeze(model.annotator_head)
            decoder_params = _unfreeze(model.decoder)
            tracked        = set(head_params + decoder_params)
            norm_params    = _unfreeze_norms(exclude=tracked)

            optimizer = optim.Adam([
                {'params': head_params,    'lr': config.lr},
                {'params': decoder_params, 'lr': config.transfer_lr},
                {'params': norm_params,    'lr': config.transfer_lr},
            ])

        case 4:
            # Phase 4: Heads + decoder + encoder last block + norms. Adds scheduler.
            _freeze_all()
            head_params      = _unfreeze(model.segmentation_head) + _unfreeze(model.annotator_head)
            decoder_params   = _unfreeze(model.decoder)
            last_block_params = _unfreeze(model.encoder.layer4)
            tracked           = set(head_params + decoder_params + last_block_params)
            norm_params       = _unfreeze_norms(exclude=tracked)

            optimizer = optim.Adam([
                {'params': head_params,        'lr': config.lr},
                {'params': decoder_params,     'lr': config.transfer_lr},
                {'params': norm_params,        'lr': config.transfer_lr},
                {'params': last_block_params,  'lr': config.transfer_lr},
            ])
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)

        case 5:
            # Phase 5: Full model unfrozen with differential LRs. Adds scheduler.
            _freeze_all()
            head_params        = _unfreeze(model.segmentation_head) + _unfreeze(model.annotator_head)
            decoder_params     = _unfreeze(model.decoder)
            last_block_params  = _unfreeze(model.encoder.layer4)
            mid_block_params   = (
                _unfreeze(model.encoder.layer3) +
                _unfreeze(model.encoder.layer2)
            )
            base_params        = (
                _unfreeze(model.encoder.layer1) +
                _unfreeze(model.encoder.layer0)
            )
            tracked    = set(head_params + decoder_params + last_block_params
                             + mid_block_params + base_params)
            norm_params = _unfreeze_norms(exclude=tracked)

            optimizer = optim.Adam([
                {'params': head_params,       'lr': config.lr},
                {'params': decoder_params,    'lr': config.transfer_lr},
                {'params': last_block_params, 'lr': config.transfer_lr},
                {'params': mid_block_params,  'lr': config.transfer_lr},
                {'params': base_params,       'lr': config.transfer_lr},
                {'params': norm_params,       'lr': config.transfer_lr},
            ])
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)

        case _:
            # Fallback: identical to phase-free mode to avoid silent misconfiguration.
            for param in model.parameters():
                param.requires_grad = True

            optimizer = optim.Adam(model.parameters(), lr=config.lr)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)

    return optimizer, scheduler