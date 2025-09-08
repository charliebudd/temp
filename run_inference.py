import os
import torch
from glob import glob
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from torchvision.transforms.functional import resize
from torchvision.utils import draw_segmentation_masks
from torchvision.io import write_png

from src.sarrarp import SarrarpDataset, CLASSES
from src.utils import miou


def main(args):

    model = torch.load(f"{args.model_directory}/best_model.pt", weights_only=False).cuda().eval()
    image_norm = Normalize(model.norm_mean, model.norm_std)

    test_videos = glob(f"{args.data_directory}/test/*")
    test_dataset = SarrarpDataset(test_videos, None, None)

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.batch_size, pin_memory=True, drop_last=False
    )

    score_fn = miou
    all_scores = []

    with torch.inference_mode():

        for frames, targets, paths in test_dataloader:

            frames, targets = frames.cuda(), targets.cuda()
            logits = model(image_norm(frames))

            predictions = torch.nn.functional.one_hot(logits.argmax(dim=1), len(CLASSES)).permute(0, 3, 1, 2)
            scores = score_fn(predictions, targets)

            all_scores += scores.cpu().tolist()

            if not args.eval_only:
                for frame, target, prediction, path in zip(frames, targets, predictions, paths):
                    frame = (frame * 255).to(torch.uint8)
                    prediction_vis = draw_segmentation_masks(torch.zeros_like(frame), prediction.bool(), alpha=1.0)
                    target_vis = draw_segmentation_masks(torch.zeros_like(frame), target.bool(), alpha=1.0)

                    visualisation = torch.cat([
                        frame, prediction_vis, target_vis
                    ], dim=1)
                    visualisation = resize(visualisation, [visualisation.size(1) // 4, visualisation.size(2) // 4])
                    os.makedirs(os.path.dirname(f"{args.model_directory}/inference/{path}"), exist_ok=True)
                    write_png(visualisation.cpu(), f"{args.model_directory}/inference/{path.replace(".png", "_vis.png")}")
                    write_png(prediction.argmax(dim=0, keepdim=True).cpu().to(torch.uint8), f"{args.model_directory}/inference/{path.replace(".png", "_pred.png")}")

    
    score = torch.tensor(all_scores).nanmean().item()
    print(f"Test mIoU: {score:0.3f}")

if __name__ == "__main__":
    parser = ArgumentParser("Training scirpt")
    parser.add_argument("--data-directory", type=str, default="sarrarp")
    parser.add_argument("--model-directory", type=str, default="trained-model")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    main(args)
