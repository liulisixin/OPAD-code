import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from transformers import CLIPVisionModelWithProjection
from util_ip_adapter import ImageProjModel
import os
from safetensors import safe_open


# Define the discriminator
class DinoDiscriminator(nn.Module):
    def __init__(self, model_name=None, num_classes=1, freeze_dino=True, resize_method='resize'):
        super(DinoDiscriminator, self).__init__()

        # Load the pretrained DINO backbone
        self.backbone = timm.create_model(
                            model_name,
                            pretrained=True,
                            num_classes=0,  # remove classifier nn.Linear
                        )
        self.backbone = self.backbone.eval()
        self.img_resolution = self.backbone.patch_embed.img_size[0]

        # Freeze DINO backbone if required
        if freeze_dino:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Define a simple classification head
        self.classification_head = nn.Sequential(
            nn.Linear(self.backbone.embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

        self.resize_method = resize_method

    def forward(self, x):
        if self.resize_method == 'resize':
            x = F.interpolate(x, self.img_resolution, mode='area')
        elif self.resize_method == 'padding':
            pad_size = int((self.img_resolution - x.size(-1))/2)
            x = F.pad(x, (pad_size, pad_size, pad_size, pad_size))
        else:
            raise NotImplementedError
        # Extract features using DINO backbone
        features = self.backbone(x)
        if isinstance(features, tuple):  # Handle multi-layer outputs
            features = features[0]

        # Classify features
        logits = self.classification_head(features)
        return logits



class IPAdapterDisc(nn.Module):
    def __init__(self, image_encoder_path, ip_ckpt, device, num_tokens=4, num_classes=2, disc_head_layer=2):
        super(IPAdapterDisc, self).__init__()

        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens

        self.cross_attention_dim = 768

        self.dtype = torch.float32

        # load image encoder
        self.img_resolution = 224
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=self.dtype
        )

        # image proj model
        self.image_proj_model = self.init_proj()
        self.load_ip_adapter_only_proj()

        self.image_encoder = self.image_encoder.eval()
        self.image_proj_model = self.image_proj_model.eval()
        self.image_encoder.requires_grad_(False)
        self.image_proj_model.requires_grad_(False)

        # Define a simple classification head
        if disc_head_layer == 2:
            self.classification_head = nn.Sequential(
                nn.Linear(768*4, 512),
                nn.ReLU(),
                nn.Linear(512, num_classes)
            )
        elif disc_head_layer == 1:
            self.classification_head = nn.Sequential(
                nn.Linear(768*4, num_classes)
            )
        else:
            raise NotImplementedError

    def init_proj(self):
        image_proj_model = ImageProjModel(
            # cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            cross_attention_dim=self.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=self.dtype)
        return image_proj_model

    def load_ip_adapter_only_proj(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
    
    def get_image_id_embeds(self, image=None):
        # clip need 224*224. In the original code, clip_image_processor will resize the pil images to 224*224
        
        clip_image_embeds = self.image_encoder(image).image_embeds

        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        return image_prompt_embeds
    
    def forward(self, x):
        x = F.interpolate(x, self.img_resolution, mode='area')
        # Extract features using DINO backbone
        features = self.get_image_id_embeds(x)

        features_flattened = features.view(features.size(0), -1)
        # Classify features
        logits = self.classification_head(features_flattened)
        return logits


class TimmBackboneDiscriminator(nn.Module):
    def __init__(self, model_name, num_classes=2, freeze_backbone=True, resize_method='resize'):
        super(TimmBackboneDiscriminator, self).__init__()

        # Load backbone from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0  # remove classifier head
        )
        self.backbone = self.backbone.eval()

        # Handle image resolution
        if hasattr(self.backbone, 'patch_embed'):
            self.img_resolution = self.backbone.patch_embed.img_size[0]
        else:
            self.img_resolution = 224  # Default fallback (e.g., ResNet)

        # Freeze backbone if required
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Classification head
        embed_dim = getattr(self.backbone, 'num_features', None)
        if embed_dim is None:
            raise ValueError(f"Cannot find 'num_features' for model {model_name}")

        self.classification_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

        self.resize_method = resize_method

    def forward(self, x):
        if self.resize_method == 'resize':
            x = F.interpolate(x, size=(self.img_resolution, self.img_resolution), mode='area')
        elif self.resize_method == 'padding':
            pad_size = int((self.img_resolution - x.size(-1)) / 2)
            x = F.pad(x, (pad_size, pad_size, pad_size, pad_size))
        else:
            raise NotImplementedError

        features = self.backbone(x)
        if isinstance(features, tuple):  # Handle multi-output
            features = features[0]

        logits = self.classification_head(features)
        return logits


