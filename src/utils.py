import torch
import random
import numpy as np


# Setting all seeds for reproducability
def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# # mean IoU implementation skipping background (assumed class 0) as
# # this can over inflate results. Not reducing over batch.
def miou(predictions: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:

    # assumes one-hot [B,C,H,W]
    predictions, targets = predictions[:,1:].bool(), targets[:,1:].bool()

    intersection = (predictions & targets).sum(dim=(2,3))
    union = (predictions | targets).sum(dim=(2,3))

    iou = intersection.float() / (union.float() + eps)

    # ignore absent classes
    valid = union > 0
    iou = torch.where(valid, iou, torch.nan)

    return torch.nanmean(iou, dim=1)
