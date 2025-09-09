### DinoV3 x SARRAP

![Example segmentation](./example.png)

I've chosen to train a simple adapter to evaluate DinoV3 in surgery. Despite the simple adpater the model performs commendably acheiving mIoU's of around 0.68 after training for just a single epoch.
Here's a break down of the approach...
- Data is split 80/20 into training and validation splits.
- ViT attention blocks are trained with LoRA to prevent overfitting.
- Dense pixel-wise logits are derived from token embeddings via a simple 3 layer MLP. 
- CrossEntropy loss is weighted per class based on pixel-wise frequencies.
- Various spatial and color augmentations applied during trianing.
- Images downsampled to half resolution to reduce memory footprint.  
- Resolution (960×520) is still high enough to preserve fine objects like thread.  
- Output logits are interpolated to full-resolution before loss and score calculations.  

NOTE: DinoV3 is a gated huggingface repo, I've provided a `hf_token.txt` file in my email which needs to be place in the root of this project to run the code.

## Data Prep
Zip archives extracted in to `sarrarp/train` and `sarrarp/test`.
Video frames extracted into `frames` dir e.g. `sarrarp/train/video_02/video_left.avi` extracted to `sarrarp/train/video_02/frames/000000000.png`
Bash scripts provided to help.

## Environment Prep
Requirements can be installed with
```
pip install -r requirements.txt
```

## Training
Training can be run with the following command with the provided token...
```
python run_training.py --data-directory <default is sarrarp> --output-directory <default is trained-models>
``` 
At this resolution, and with DinoV3 patch size (16x16) there are a large number of tokens. Training with the default config that requires ~30Gb of VRAM. To run on a smaller GPU use the `--low-res` flag to run at quater resolution.


## Inference
Inference on the test data split can be run with...
```
python run_inference.py --data-directory <default is sarrarp> --model-directory <default is trained-models>
```
This will produce visualisations images (rgb, prediction, groundtruth) and prediction masks saved into the model directory e.g. `trained-model/inference/video_41/frames/000000000_pred.png`. Evaluation is also performed during this by calculating mIoU over all samples and averaging. To run only the evaluation use the `--eval-only` flag.

## Improvemnts
Improvements would likely come from better adapter architecture, possibly incorporating skip connection from earlier in the ViT to help guide final segmentation output. Naturally running at full resolution would help however this would require splitting across multiple GPUs and I feel I already have an advantage having a 40GB GPU to hand. I have not explored class-wise scores which would be important for assessing how to improve the model, however it is clear that the smaller objects have the most errors. Again, running at higher resolution would help.
