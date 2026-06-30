from transformers import PretrainedConfig, PreTrainedModel, CLIPImageProcessor
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import models as torchvision_models
from torchvision import models
import torch
import os
import sys
import ipdb


class ResNeXtImageProcessor(CLIPImageProcessor):
    def __init__(self, image_size=224, image_mean=None, image_std=None, **kwargs):
        super().__init__(**kwargs)
        self.size = {"height": image_size, "width": image_size}
        self.image_mean = image_mean or [0.485, 0.456, 0.406]
        self.image_std = image_std or [0.229, 0.224, 0.225]
        self.do_center_crop = False
        self.do_resize = True
        self.do_normalize = True
        self.resample = Image.BILINEAR


class ResNeXtVisionTower(torch.nn.Module):
    def __init__(self, vision_tower, args, delay_load=False, **kwargs):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.trainable = getattr(args, "tune_vision_tower", False)
        self.hidden_size = None
        self.local_rank = kwargs.get('local_rank', None)
        self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        # Image processor
        self.image_processor = ResNeXtImageProcessor()

        # Vision tower initialization
        # device_map = f"cuda:{self.local_rank}" if self.local_rank is not None else "cuda" if torch.cuda.is_available() else "cpu"
        self.vision_tower = torchvision_models.__dict__["resnext50_32x4d"]()
        # self.vision_tower = models.resnext50_32x4d()
        
        # self.vision_tower = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        self.hidden_size = self.vision_tower.fc.in_features
        self.vision_tower.fc = torch.nn.Identity()
        
        # Checkpoint loading
        # print("***************************************")
        # print(self.vision_tower_name)
        if self.vision_tower_name is None:
            state_dict = None
        elif self.vision_tower_name == "resnext_scratch":
            state_dict = None
        elif os.path.exists(self.vision_tower_name):
            checkpoint = self.vision_tower_name
            state_dict = torch.load(checkpoint, map_location="cpu")['teacher']
            print("Take key teacher in local checkpoint dict")
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            # remove `encoder.` prefix if it exists
            state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
        elif "eminorhan" in self.vision_tower_name:
            if not os.path.exists(self.vision_tower_name):
                checkpoint = hf_hub_download(repo_id=self.vision_tower_name, filename=self.vision_tower_name.split('/')[-1] + ".pth")
            else:
                checkpoint = os.path.join(self.vision_tower_name, self.vision_tower_name.split('/')[-1]+".pth")
            state_dict = torch.load(checkpoint, map_location="cpu")
            checkpoint_key = "teacher"
            if checkpoint_key in state_dict:
                print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            # remove `encoder.` prefix if it exists
            state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
        elif "multimodal-baby/" in self.vision_tower_name:
            assert os.path.exists(self.vision_tower_name), f"local vision backbone {self.vision_tower_name} does not exist"
            checkpoint = os.path.join(self.vision_tower_name, "last.ckpt")
            state_dict = torch.load(checkpoint, map_location="cpu")['state_dict']
            state_dict = {k.replace("vision_encoder.model.", ""): v for k, v in state_dict.items() if "vision_encoder.model." in k}
        else:
            raise ValueError(f'Unknown vision tower: {self.vision_tower_name}')
        if state_dict:
            msg = self.vision_tower.load_state_dict(state_dict, strict=False, assign=True)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(checkpoint, msg))
        else:
            import warnings
            warnings.warn('No pretrained weights specified for ResNeXt backbone, using random initialization.')

        if self.trainable:
            self.vision_tower.requires_grad_(True)
        else:
            self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    # def feature_select(self, image_forward_outs):
    #     image_features = image_forward_outs.hidden_states[self.select_layer]
    #     if self.select_feature == 'patch':
    #         image_features = image_features[:, 1:]
    #     elif self.select_feature == 'cls_patch':
    #         image_features = image_features
    #     else:
    #         raise ValueError(f'Unexpected select feature: {self.select_feature}')
    #     return image_features

    # @torch.no_grad()
    def forward(self, images):
        images = images.to(device=self.device, dtype=self.dtype)
        if self.trainable:
            image_features = self._forward_impl(images)
        else:
            with torch.no_grad():
                image_features = self._forward_impl(images)
        return image_features
    
    def _forward_impl(self, images):
        # Cannot directly call resnext.forward here, as we don't want the output to be average pooled
        outputs = self.vision_tower.conv1(images)
        outputs = self.vision_tower.bn1(outputs)
        outputs = self.vision_tower.relu(outputs)
        outputs = self.vision_tower.maxpool(outputs)
        outputs = self.vision_tower.layer1(outputs)
        outputs = self.vision_tower.layer2(outputs)
        outputs = self.vision_tower.layer3(outputs)
        outputs = self.vision_tower.layer4(outputs)
        # outputs = self.vision_tower.avgpool(outputs)
        # outputs = torch.flatten(outputs, 1)
        # outputs = self.vision_tower.fc(outputs)
        outputs = torch.flatten(outputs, start_dim=2).permute(0, 2, 1)
        return outputs

    # @property
    # def dummy_feature(self):
    #     return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype
        # return self.vision_tower.dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device
        # return self.vision_tower.device

    # @property
    # def config(self):
    #     if self.is_loaded:
    #         return self.vision_tower.config
    #     else:
    #         return self.cfg_only

    # @property
    # def hidden_size(self):
    #     return self.config.hidden_size

    # @property
    # def num_patches_per_side(self):
    #     return self.config.image_size // self.config.patch_size

    # @property
    # def num_patches(self):
    #     return (self.config.image_size // self.config.patch_size) ** 2


class CVCLVisionTower(ResNeXtVisionTower):
    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        # Image processor
        self.image_processor = ResNeXtImageProcessor()

        # Vision tower initialization
        # device_map = f"cuda:{self.local_rank}" if self.local_rank is not None else "cuda" if torch.cuda.is_available() else "cpu"
        self.vision_tower = torchvision_models.__dict__["resnext50_32x4d"]()
        self.hidden_size = self.vision_tower.fc.in_features
        self.vision_tower.fc = torch.nn.Identity()

        # Checkpoint loading
        if not os.path.exists(self.vision_tower_name):
            checkpoint = hf_hub_download(repo_id=self.vision_tower_name, filename=self.vision_tower_name.split('/')[-1]+".pth")
        else:
            checkpoint = os.path.join(self.vision_tower_name, self.vision_tower_name.split('/')[-1]+".pth")
        state_dict = torch.load(checkpoint, map_location="cpu")
        checkpoint_key = "teacher"
        if checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        # remove `encoder.` prefix if it exists
        state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
        # global local_rank
        msg = self.vision_tower.load_state_dict(state_dict, strict=False, assign=True)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(checkpoint, msg))

        if self.trainable:
            self.vision_tower.requires_grad_(True)
        else:
            self.vision_tower.requires_grad_(False)

        self.is_loaded = True




############################################################################################################
# Below is useless
############################################################################################################
# class ResNeXtConfig(PretrainedConfig):
#     model_type = "resnext50_32x4d"

#     def __init__(
#         self,
#         # model_name="dino_sfp_resnext50",
#         **kwargs,
#         ):
#         # self.model_name = model_name
#         super().__init__(**kwargs)


# class ResNeXtModel(PreTrainedModel):
#     config_class = ResNeXtConfig

#     def __init__(self, config):
#         super().__init__(config)
#         self.config = config
#         self.resnext = torchvision_models.__dict__["resnext50_32x4d"]()
#         self.resnext.fc = torch.nn.Identity()

#     @classmethod
#     def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
#         """
#         Loads the model from a pre-trained model file or directory.

#         Args:
#             pretrained_model_name_or_path (`str` or `os.PathLike`):
#                 Can be either:
#                     - A string, the *model id* of a pretrained model hosted
#                       inside a model repo on huggingface.co.
#                     - A path to a *directory* containing model weights saved using
#                       `save_pretrained`, e.g., `./my_model_directory/`.
#                     - A path or URL to a saved model file, e.g., `./my_model_directory/pytorch_model.bin`.
#         """
#         config = kwargs.pop('config', None)
#         state_dict = None
#         # Try to load config
#         if config is None:
#             try:
#                 config = cls.config_class.from_pretrained(pretrained_model_name_or_path)
#             except Exception as e:
#                 print(f"ResNeXt config not found at {pretrained_model_name_or_path}. Using default config. Error: {e}")
#                 config = cls.config_class(**kwargs)
        
#         # Initialize model
#         model = cls(config)

#         # Load state_dict from eminorhan's hub
#         checkpoint = hf_hub_download(repo_id=pretrained_model_name_or_path, filename=pretrained_model_name_or_path.split('/')[-1]+".pth")
#         cls.load_model_weights(model.resnext, checkpoint, "teacher")

#     @staticmethod
#     def load_model_weights(model, pretrained_weights, checkpoint_key):
#         if os.path.isfile(pretrained_weights):
#             state_dict = torch.load(pretrained_weights, map_location="cpu")
#             if checkpoint_key is not None and checkpoint_key in state_dict:
#                 print(f"Take key {checkpoint_key} in provided checkpoint dict")
#                 state_dict = state_dict[checkpoint_key]

#             # remove `module.` prefix
#             state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
#             # remove `backbone.` prefix induced by multicrop wrapper
#             state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
#             # remove `encoder.` prefix if it exists
#             state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}

#             msg = model.load_state_dict(state_dict, strict=False)
#             print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
#         else:
#             print("There is no reference weights available for this model => We use random weights.")

#     def forward(self, pixel_values, return_dict=True):
#         # Cannot directly call resnext.forward here, as we don't want the output to be average pooled
#         outputs = self.resnext.conv1(pixel_values)
#         outputs = self.resnext.bn1(outputs)
#         outputs = self.resnext.relu(outputs)
#         outputs = self.resnext.maxpool(outputs)
#         outputs = self.resnext.layer1(outputs)
#         outputs = self.resnext.layer2(outputs)
#         outputs = self.resnext.layer3(outputs)
#         outputs = self.resnext.layer4(outputs)
#         # outputs = self.resnext.avgpool(outputs)
#         # outputs = torch.flatten(outputs, 1)
#         # outputs = self.resnext.fc(outputs)
#         if not return_dict:
#             return (outputs,)
#         return {"last_hidden_state": outputs}


if __name__ == "__main__":
    # model = ResNeXtVisionTower(vision_tower="eminorhan/dino_sfp_resnext50", args=None)
    model = ResNeXtVisionTower(vision_tower="wkvong/cvcl_s_dino_resnext50_embedding", args=None)
    print("Success!")
    dummy_input = torch.randn(1, 3, 224, 224)
    print(model(dummy_input).shape)