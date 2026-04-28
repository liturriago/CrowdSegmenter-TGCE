import torch
import numpy as np
import random
import os

def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility across different libraries.

    This function fixes seeds for Python's random module, NumPy, and PyTorch 
    (including CUDA). It also configures CuDNN for deterministic behavior.

    Args:
        seed (int): The seed value to use for all random number generators.
    """
    # 1. Basic seeds for Python and NumPy
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    # 2. PyTorch seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    # 3. CuDNN configuration for determinism
    # Note: This may slightly reduce performance but ensures consistency
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Seeds fixed at: {seed}")