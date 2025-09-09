import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from transformers import AutoModel, AutoImageProcessor
from peft import LoraConfig, get_peft_model

# Adapter for DinoV3 ViTs from huggingface...
# https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
# https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
# https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m

# The model applies a simple MLP to the token embeddings to derive dense logit maps.
# PEFT is optionally used to provide LoRA fine tuning (allthough this provides the best performance).
class DinoV3ViT(torch.nn.Module):
    def __init__(self, num_classes:int, backbone_size:str, backbone_training_mode:str, lora_rank:int, low_res:bool):
        super().__init__()
        self.num_classes = num_classes
        self.low_res = low_res

        hf_model_name = f"facebook/dinov3-{backbone_size}16-pretrain-lvd1689m"
        self.model = AutoModel.from_pretrained(hf_model_name)

        if backbone_training_mode == "frozen":
            for p in self.model.parameters():
                p.requires_grad = False
        elif backbone_training_mode == "lora":
            peft_cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank,
                target_modules="all-linear",
                lora_dropout=0.1,
                bias="none",
            )
            self.model = get_peft_model(self.model, peft_cfg)
        elif backbone_training_mode == "full":
            pass
        else:
            raise NotImplementedError(f"Backbone trianing mode \"{backbone_training_mode}\" not implemented.")
        
        processor = AutoImageProcessor.from_pretrained(hf_model_name)
        self.input_size = (processor.size["height"],  processor.size["width"])
        self.norm_mean = processor.image_mean
        self.norm_std = processor.image_std

        output_feature_dim = num_classes * (self.model.config.patch_size) ** 2
        self.seg_head = torch.nn.Sequential(
            torch.nn.Linear(self.model.config.hidden_size, output_feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(output_feature_dim, output_feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(output_feature_dim, output_feature_dim),
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:

        if self.low_res:
            input = torch.nn.functional.interpolate(x, scale_factor=0.25, mode="bilinear")
        else:
            input = torch.nn.functional.interpolate(x, scale_factor=0.5, mode="bilinear")
        
        outputs = self.model(input)

        num_skip_tokens = 1 + self.model.config.num_register_tokens
        patch_features = outputs.last_hidden_state[:, num_skip_tokens:, :]

        patch_logits = self.seg_head(patch_features)

        B, N, _ = patch_logits.shape
        P = self.model.config.patch_size
        H, W = input.shape[-2] // P, input.shape[-1] // P
        patch_logits = patch_logits.view(B, N, self.num_classes, P, P)
        pixel_logits = rearrange(patch_logits, "b (h w) c ph pw -> b c (h ph) (w pw)", h=H, w=W)

        if pixel_logits.shape[-2:] != x.shape[-2:]:
            pixel_logits = torch.nn.functional.interpolate(pixel_logits, x.shape[-2:], mode="bilinear")

        return pixel_logits
