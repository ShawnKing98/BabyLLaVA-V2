import torch
import timm
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import models as torchvision_models
from transformers import CLIPImageProcessor
import os
import sys
import re
from collections import OrderedDict
from types import SimpleNamespace

_RULES = [
    # top-level
    (r"^backbone\.cls_token$",                        "class_token"),
    (r"^backbone\.pos_embed$",                        "encoder.pos_embedding"),
    (r"^backbone\.patch_embed\.proj\.",               "conv_proj."),
    (r"^backbone\.norm\.",                            "encoder.ln."),

    # inside each transformer block i = 0‥23
    (r"^backbone\.blocks\.(\d+)\.norm1\.",            r"encoder.layers.encoder_layer_\1.ln_1."),
    (r"^backbone\.blocks\.(\d+)\.norm2\.",            r"encoder.layers.encoder_layer_\1.ln_2."),
    (r"^backbone\.blocks\.(\d+)\.mlp\.fc1\.",         r"encoder.layers.encoder_layer_\1.mlp.0."),
    (r"^backbone\.blocks\.(\d+)\.mlp\.fc2\.",         r"encoder.layers.encoder_layer_\1.mlp.3."),
    (r"^backbone\.blocks\.(\d+)\.attn\.proj\.",       r"encoder.layers.encoder_layer_\1.self_attention.out_proj."),

    # *** fixed: weight / bias done explicitly ***
    (r"^backbone\.blocks\.(\d+)\.attn\.qkv\.weight$", r"encoder.layers.encoder_layer_\1.self_attention.in_proj_weight"),
    (r"^backbone\.blocks\.(\d+)\.attn\.qkv\.bias$",   r"encoder.layers.encoder_layer_\1.self_attention.in_proj_bias"),
]

def state_dict_convert(hf_state):
    """Rename ViT-L/16 timm/HF keys → torchvision keys (drop head.*)."""
    tv_state = OrderedDict()
    for k, v in hf_state.items():
        if k.startswith("head."):        # skip classifier head
            continue
        for pat, repl in _RULES:
            k = re.sub(pat, repl, k)
        tv_state[k] = v
    return tv_state


def should_use_timm_vit(state_dict):
    return any(key.endswith(".ls1.gamma") for key in state_dict)


def state_dict_convert_dinov2(state_dict):
    timm_state = OrderedDict()
    for key, value in state_dict.items():
        if not key.startswith("backbone."):
            continue
        converted_key = key[len("backbone."):]
        if converted_key in {"mask_token", "register_tokens"}:
            continue
        timm_state[converted_key] = value
    return timm_state

    
class ViTImageProcessor(CLIPImageProcessor):
    def __init__(self, image_size=224, image_mean=None, image_std=None, **kwargs):
        super().__init__(**kwargs)
        self.size = {"height": image_size, "width": image_size}
        self.image_mean = image_mean or [0.485, 0.456, 0.406]
        self.image_std = image_std or [0.229, 0.224, 0.225]
        self.do_center_crop = False
        self.do_resize = True
        self.do_normalize = True
        self.resample = Image.BILINEAR

class ViTVisionTower(torch.nn.Module):
    def __init__(self, vision_tower, args, delay_load=False, **kwargs):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.trainable = getattr(args, "tune_vision_tower", False)
        self.hidden_size = None
        self.image_size = 224
        self.patch_size = 16
        self.uses_timm_vit = False
        self.local_rank = kwargs.get('local_rank', None)
        self.load_model()

    def _build_torchvision_vit(self):
        self.uses_timm_vit = False
        self.vision_tower = torchvision_models.__dict__["vit_l_16"](weights=None)
        self.hidden_size = self.vision_tower.heads.head.in_features
        self.vision_tower.heads.head = torch.nn.Identity()

    def _build_timm_vit(self):
        self.uses_timm_vit = True
        self.vision_tower = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            num_classes=0,
            init_values=1e-5,
            img_size=self.image_size,
        )
        self.hidden_size = self.vision_tower.embed_dim

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f'{self.vision_tower_name} is already loaded, `load_model` called again, skipping.')
            return
        self.image_processor = ViTImageProcessor(image_size=224)

        # Checkpoint loading
        if self.vision_tower_name == "vit_scratch":
            state_dict = None
        elif os.path.exists(self.vision_tower_name):
            checkpoint = self.vision_tower_name
            assert self.vision_tower_name.endswith('.pth'), f"the passed in vision backbone checkpoint ({checkpoint}) doesn't end with '.pth'."
            state_dict = torch.load(checkpoint, map_location="cpu")
        elif "eminorhan" in self.vision_tower_name:
            if not os.path.exists(self.vision_tower_name):
                checkpoint = hf_hub_download(repo_id=self.vision_tower_name, filename="dino_say_vitl16.pth")
            else:
                checkpoint = os.path.join(self.vision_tower_name, "dino_say_vitl16.pth")
            state_dict = torch.load(checkpoint, map_location="cpu")
        else:
            raise ValueError(f'Unknown vision tower: {self.vision_tower_name}')
        if state_dict:
            checkpoint_key = "teacher"
            if checkpoint_key in state_dict:
                print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]

            if should_use_timm_vit(state_dict):
                self._build_timm_vit()
                state_dict = state_dict_convert_dinov2(state_dict)
            else:
                self._build_torchvision_vit()
                state_dict = state_dict_convert(state_dict)

            msg = self.vision_tower.load_state_dict(state_dict, strict=False, assign=True)
            print(f'Pretrained weights found at {checkpoint} and loaded with msg: {msg}')
        else:
            import warnings
            self._build_torchvision_vit()
            warnings.warn('No pretrained weights specified for ViT backbone, using random initialization.')

        if self.trainable:
            self.vision_tower.requires_grad_(True)
        else:
            self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def forward(self, images):
        images = images.to(device=self.device, dtype=self.dtype)
        if self.trainable:
            image_features = self._forward_impl(images)
        else:
            with torch.no_grad():
                image_features = self._forward_impl(images)
        return image_features

    def _forward_impl(self, images):
        if self.uses_timm_vit:
            return self.vision_tower.forward_features(images)

        # Reshape and permute the input tensor
        x = self.vision_tower._process_input(images)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.vision_tower.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.vision_tower.encoder(x)

        # Classifier "token" as used by standard language architectures
        # x = x[:, 0]

        # x = self.heads(x)

        return x

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def config(self):
        return SimpleNamespace(
            hidden_size=self.hidden_size,
            image_size=self.image_size,
            patch_size=self.patch_size,
        )

    @property
    def num_patches_per_side(self):
        return self.image_size // self.patch_size

    @property
    def num_patches(self):
        return self.num_patches_per_side ** 2

if __name__ == "__main__":
    vit_tower = ViTVisionTower("eminorhan/dino_say_vitl16", args=None)
    print(vit_tower.device)
    print(vit_tower.dtype)
    dummy_input = torch.randn(1, 3, 224, 224)
    print(vit_tower(dummy_input).shape)
