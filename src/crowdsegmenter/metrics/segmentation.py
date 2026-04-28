import torch

def dice_coefficient(predictions: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Computes the Dice Coefficient for segmentation.
    
    Args:
        predictions (torch.Tensor): Binary predictions (0 or 1), shape (N, C, H, W).
        targets (torch.Tensor): Binary targets (0 or 1), shape (N, C, H, W).
        smooth (float): Smoothing factor to prevent division by zero.
        
    Returns:
        torch.Tensor: Dice coefficient value.
    """
    # Flatten predictions and targets
    preds_flat = predictions.view(-1)
    targets_flat = targets.view(-1)
    
    intersection = (preds_flat * targets_flat).sum()
    union = preds_flat.sum() + targets_flat.sum()
    
    return (2. * intersection + smooth) / (union + smooth)

def iou_score(predictions: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Computes the Intersection over Union (IoU) for segmentation.
    
    Args:
        predictions (torch.Tensor): Binary predictions (0 or 1), shape (N, C, H, W).
        targets (torch.Tensor): Binary targets (0 or 1), shape (N, C, H, W).
        smooth (float): Smoothing factor to prevent division by zero.
        
    Returns:
        torch.Tensor: IoU score.
    """
    preds_flat = predictions.view(-1)
    targets_flat = targets.view(-1)
    
    intersection = (preds_flat * targets_flat).sum()
    union = preds_flat.sum() + targets_flat.sum() - intersection
    
    return (intersection + smooth) / (union + smooth)
