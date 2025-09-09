import os
import json
import torch
import huggingface_hub
from glob import glob
from tqdm import tqdm
from itertools import islice
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomAffine, RandomHorizontalFlip
from torchvision.transforms import ColorJitter, Normalize, InterpolationMode

from src.sarrarp import SarrarpDataset, CLASSES
from src.models import DinoV3ViT
from src.utils import set_seed, miou


def main(args):

    # Seeding for reproducability
    set_seed(args.seed)

    # Output directory for saving models & info
    os.makedirs(args.output_directory, exist_ok=True)

    # Backbone:
    # I'm using a dinov3 backbone mainly because I havent had a chance to try them
    # yet and wanted to play with it. See class definition for adapter implementaion.
    print(f"Loading backbone...")
    model = DinoV3ViT(
        num_classes=len(CLASSES),
        backbone_size=args.backbone_size,
        backbone_training_mode=args.backbone_training,
        lora_rank=args.lora_rank,
        low_res=args.low_res,
    ).cuda().train()

    # Augmentation/Transforms:
    # Simple augmentations to increase variability, spatial tranform
    # is applied to image and target, color just to the image.

    image_norm = Normalize(model.norm_mean, model.norm_std)

    spatial_augmentation = Compose([
        RandomAffine(
            degrees=15.0,
            translate=(0.2, 0.2),
            scale=(0.7, 1.3),
            shear=10.0,
            interpolation=InterpolationMode.BILINEAR
        ),
        RandomHorizontalFlip(p=0.5),
    ])

    color_augmentation = Compose([
        ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
    ])

    # Data:
    # Train-val split done along video boundary to prevent data leakage.
    # Using workers to reduce trainnig process waiting for data loading.
    # Drop last during training to prevent an over biased gradient step.
    print(f"Setting up data loaders...")
    videos = glob(f"{args.data_directory}/train/*")
    num_val_samples = max(round(args.validation_split_size * len(videos)), 1)
    training_videos, validation_videos = videos[:-num_val_samples], videos[-num_val_samples:]

    train_dataset = SarrarpDataset(training_videos, spatial_augmentation, Compose([color_augmentation, image_norm]))
    validation_dataset = SarrarpDataset(validation_videos, None, image_norm)

    print(f"Train Samples: {len(train_dataset)}")
    print(f"Val Samples: {len(validation_dataset)}")
    print("")

    training_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=16, pin_memory=True, drop_last=True
    )
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=16, pin_memory=True, drop_last=False
    )

    # Optim:
    # I'm using a weighted cross entropy to account for class imbalance.
    # Standard Adam optimiser with a fixed gloabl learnign rate. Could 
    # still add learning rate scheduling/decay but not so important
    # with Adam as it already adapts the gradient updates.
    print("Estimating class weights...")
    n = 100
    class_freq= torch.stack([
        target.cuda().mean(dim=[0, 2, 3]) for _, target, _ in tqdm(islice(training_dataloader, n), total=n)
    ]).mean(dim=0)
    print(list(zip(CLASSES, class_freq.cpu().tolist())))
    print("")

    loss_fn = torch.nn.CrossEntropyLoss(1 - class_freq)
    optimiser = torch.optim.Adam(model.parameters(), args.learning_rate)
    score_fn = miou


    # Training Loop:
    # Simple training loop with validation run every so often.
    # Early stopping if validation result fails to improve.

    step_count = 0
    early_stop_timer = 0
    running_losses = []

    training_losses = []
    validation_scores = []

    for _ in range(args.max_epochs):

        for frames, targets, _ in training_dataloader:

            frames, targets = frames.cuda(), targets.cuda()
            logits = model(frames)

            loss = loss_fn(logits, targets)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            running_losses.append(loss.item())
            step_count += 1

            if step_count % args.validation_steps == 0:

                model.eval()

                with torch.no_grad():

                    all_scores = []
                    for frames, targets, _ in validation_dataloader:
                        frames, targets = frames.cuda(), targets.cuda()
                        logits = model(frames)
                        predictions = torch.nn.functional.one_hot(logits.argmax(dim=1), len(CLASSES)).permute(0, 3, 1, 2) # One hot predicted class indices
                        scores = score_fn(predictions, targets) # can return nan
                        all_scores += scores.cpu().tolist()
                model.train()

                training_loss = sum(running_losses) / len(running_losses)
                training_losses.append(training_loss)
                running_losses = []
                
                validation_score = torch.tensor(all_scores).nanmean().item()
                validation_scores.append(validation_score)

                # Logging and early stopping:
                # In a more formal setup I would use something like W&B for logging, and 
                # would save more info such as git commit id, args passed to the script, etc.
                # For this I'm just saving training losses and validation scores to files.
                print(f"Mini Epoch {len(training_losses)-1: 3d}: mean loss {training_loss:0.3f}, score {validation_score:0.3f}")

                with open(f"{args.output_directory}/metrics.json", "w") as f:
                    json.dump({
                        "training_losses": training_losses,
                        "validation_scores": validation_scores,
                    }, f)

                if validation_score == max(validation_scores):
                    early_stop_timer = 0
                    torch.save(model, f"{args.output_directory}/best_model.pt")
                else:
                    early_stop_timer += 1
                    if early_stop_timer > args.early_stop_patience:
                        print("Early stop patience reached.")
                        return


if __name__ == "__main__":
    parser = ArgumentParser("Training scirpt")
    parser.add_argument("--data-directory", type=str, default="sarrarp")
    parser.add_argument("--output-directory", type=str, default="trained-model")
    parser.add_argument("--backbone-size", type=str, choices=["vits", "vitb", "vitl"], default="vitl")
    parser.add_argument("--backbone-training", type=str, choices=["frozen", "lora", "full"], default="lora")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--low-res", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--validation-split-size", type=float, default=0.2)
    parser.add_argument("--validation-steps", type=int, default=500)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=645486)
    args = parser.parse_args()
    
    # I've provided an access token with sufficient permisions.
    with open("hf_token.txt") as file:
        token = file.read()
    huggingface_hub.login(token)

    main(args)
