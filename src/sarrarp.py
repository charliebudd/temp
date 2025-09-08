import torch
from glob import glob
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from typing import Callable, List


CLASSES = [
    "Background",
    "ToolClasper",
    "ToolWrist",
    "ToolShaft",
    "SuturingNeedle",
    "Thread",
    "SuctionTool",
    "NeedleHolder",
    "Clamps",
    "Catheter",
]


# Simple dataset using glob to match image paths.
# Spatial transformation applied jointly to frame and target.
# Color transformation applied to frame only.
class SarrarpDataset(Dataset):
    def __init__(self, video_directories:List[str], spatial_transform:Callable=None, colour_transform:Callable=None):
        self.frame_paths = sum([glob(f"{dir}/frames/*.png") for dir in video_directories], [])
        self.seg_paths = [p.replace("frames", "segmentation") for p in self.frame_paths]
        self.spatial_transform = spatial_transform
        self.colour_transform = colour_transform

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        sample_path = "/".join(self.frame_paths[idx].split("/")[-3:])
        frame = read_image(self.frame_paths[idx], ImageReadMode.RGB).float() / 255.0
        target = read_image(self.seg_paths[idx], ImageReadMode.GRAY).long().squeeze()
        target = torch.nn.functional.one_hot(target.long(), len(CLASSES)).float().permute(2, 0, 1)
        if self.spatial_transform:
            stack = torch.cat([frame, target])
            stack = self.spatial_transform(stack)
            frame, target = stack[:3], stack[3:]
        if self.colour_transform:
            frame = self.colour_transform(frame)
        return frame, target, sample_path
