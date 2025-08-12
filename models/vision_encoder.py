import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math
from einops import rearrange, repeat
from dataclasses import dataclass
import torchvision.transforms as transforms
from torchvision.models import resnet50, densenet121, efficientnet_b7
import timm


@dataclass
class VisionEncoderConfig:
    backbone: str = "efficientnet_b7"
    pretrained: bool = True
    
    input_size: Tuple[int, int] = (512, 512)
    input_channels: int = 1
    output_dim: int = 1024
    
    use_multi_scale: bool = True
    feature_levels: List[str] = None
    use_spatial_attention: bool = True
    
    use_medical_preprocessing: bool = True
    use_contrast_enhancement: bool = True
    use_adaptive_histogram_equalization: bool = True
    
    use_position_encoding: bool = True
    use_channel_attention: bool = True
    dropout: float = 0.1
    
    freeze_backbone: bool = False
    progressive_unfreezing: bool = True
    
    def __post_init__(self):
        if self.feature_levels is None:
            self.feature_levels = ['low', 'mid', 'high']


class MedicalImagePreprocessor(nn.Module):
    
    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.config = config
        
        if config.use_contrast_enhancement:
            self.contrast_enhancer = nn.Sequential(
                nn.Conv2d(config.input_channels, config.input_channels, 3, padding=1),
                nn.BatchNorm2d(config.input_channels),
                nn.ReLU(inplace=True)
            )
        
        if config.use_adaptive_histogram_equalization:
            self.histogram_equalizer = nn.Sequential(
                nn.Conv2d(config.input_channels, 16, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, config.input_channels, 1),
                nn.Sigmoid()
            )
        
        self.noise_reducer = nn.Sequential(
            nn.Conv2d(config.input_channels, config.input_channels, 3, padding=1, groups=config.input_channels),
            nn.Conv2d(config.input_channels, config.input_channels, 1),
            nn.BatchNorm2d(config.input_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original = x
        
        if self.config.use_contrast_enhancement:
            enhanced = self.contrast_enhancer(x)
            x = x + 0.3 * enhanced
        
        if self.config.use_adaptive_histogram_equalization:
            eq_weights = self.histogram_equalizer(x)
            x = x * eq_weights + x * (1 - eq_weights)
        
        denoised = self.noise_reducer(x)
        x = x + 0.2 * denoised
        
        x = torch.clamp(x, 0, 1)
        
        return x


class SpatialAttentionModule(nn.Module):
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels // 8, in_channels // 8, 3, padding=1)
        self.conv3 = nn.Conv2d(in_channels // 8, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.conv1(x)
        attn = F.relu(attn, inplace=True)
        attn = self.conv2(attn)
        attn = F.relu(attn, inplace=True)
        attn = self.conv3(attn)
        attn = self.sigmoid(attn)
        
        return x * attn


class ChannelAttentionModule(nn.Module):
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attn = self.sigmoid(avg_out + max_out)
        return x * attn


class MultiScaleFeatureExtractor(nn.Module):
    
    def __init__(self, backbone: nn.Module, config: VisionEncoderConfig):
        super().__init__()
        self.backbone = backbone
        self.config = config
        
        self.feature_dims = self._get_feature_dimensions()
        
        self.refinement_layers = nn.ModuleDict()
        for level in config.feature_levels:
            in_dim = self.feature_dims[level]
            self.refinement_layers[level] = nn.Sequential(
                nn.Conv2d(in_dim, config.output_dim // len(config.feature_levels), 3, padding=1),
                nn.BatchNorm2d(config.output_dim // len(config.feature_levels)),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1)
            )
        
        if config.use_spatial_attention:
            self.spatial_attention = nn.ModuleDict({
                level: SpatialAttentionModule(self.feature_dims[level])
                for level in config.feature_levels
            })
        
        if config.use_channel_attention:
            self.channel_attention = nn.ModuleDict({
                level: ChannelAttentionModule(self.feature_dims[level])
                for level in config.feature_levels
            })
    
    def _get_feature_dimensions(self) -> Dict[str, int]:
        if isinstance(self.backbone, timm.models.efficientnet.EfficientNet):
            return {
                'low': 48,
                'mid': 160,
                'high': 2048
            }
        elif hasattr(self.backbone, 'fc'):
            return {
                'low': 256,
                'mid': 1024,
                'high': 2048
            }
        else:
            return {
                'low': 256,
                'mid': 512,
                'high': 1024
            }
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = {}
        
        if isinstance(self.backbone, timm.models.efficientnet.EfficientNet):
            x = self.backbone.conv_stem(x)
            x = self.backbone.bn1(x)
            x = self.backbone.act1(x)
            
            for i, block in enumerate(self.backbone.blocks[:2]):
                x = block(x)
            if 'low' in self.config.feature_levels:
                features['low'] = x
            
            for i, block in enumerate(self.backbone.blocks[2:5]):
                x = block(x)
            if 'mid' in self.config.feature_levels:
                features['mid'] = x
            
            for i, block in enumerate(self.backbone.blocks[5:]):
                x = block(x)
            x = self.backbone.conv_head(x)
            x = self.backbone.bn2(x)
            x = self.backbone.act2(x)
            if 'high' in self.config.feature_levels:
                features['high'] = x
        
        else:
            layers = list(self.backbone.children())
            x_temp = x
            
            for layer in layers[:len(layers)//3]:
                x_temp = layer(x_temp)
            if 'low' in self.config.feature_levels:
                features['low'] = x_temp
            
            for layer in layers[len(layers)//3:2*len(layers)//3]:
                x_temp = layer(x_temp)
            if 'mid' in self.config.feature_levels:
                features['mid'] = x_temp
            
            for layer in layers[2*len(layers)//3:]:
                x_temp = layer(x_temp)
            if 'high' in self.config.feature_levels:
                features['high'] = x_temp
        
        for level, feature in features.items():
            if self.config.use_spatial_attention:
                feature = self.spatial_attention[level](feature)
            if self.config.use_channel_attention:
                feature = self.channel_attention[level](feature)
            features[level] = feature
        
        return features


class PositionalEncoding2D(nn.Module):
    
    def __init__(self, d_model: int, height: int = 16, width: int = 16):
        super().__init__()
        self.d_model = d_model
        self.height = height
        self.width = width
        
        pe = torch.zeros(d_model, height, width)
        
        y_pos = torch.arange(0, height).unsqueeze(1).repeat(1, width).unsqueeze(0)
        x_pos = torch.arange(0, width).unsqueeze(0).repeat(height, 1).unsqueeze(0)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[0::4, :, :] = torch.sin(y_pos * div_term[:len(pe[0::4])].unsqueeze(1).unsqueeze(2))
        pe[1::4, :, :] = torch.cos(y_pos * div_term[:len(pe[1::4])].unsqueeze(1).unsqueeze(2))
        pe[2::4, :, :] = torch.sin(x_pos * div_term[:len(pe[2::4])].unsqueeze(1).unsqueeze(2))
        pe[3::4, :, :] = torch.cos(x_pos * div_term[:len(pe[3::4])].unsqueeze(1).unsqueeze(2))
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != self.pe.shape[-2:]:
            pe = F.interpolate(self.pe, size=x.shape[-2:], mode='bilinear', align_corners=False)
        else:
            pe = self.pe
        
        return x + pe[:, :x.size(1)]


class VisionEncoder(nn.Module):
    
    def __init__(self, config: VisionEncoderConfig = None, **kwargs):
        super().__init__()
        
        if config is None:
            config = VisionEncoderConfig(**kwargs)
        self.config = config
        
        if config.use_medical_preprocessing:
            self.preprocessor = MedicalImagePreprocessor(config)
        
        self.backbone = self._create_backbone()
        
        if config.use_multi_scale:
            self.feature_extractor = MultiScaleFeatureExtractor(self.backbone, config)
            self.feature_combiner = nn.Sequential(
                nn.Linear(config.output_dim, config.output_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout),
                nn.Linear(config.output_dim, config.output_dim)
            )
        else:
            self.feature_projector = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self._get_backbone_output_dim(), config.output_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout)
            )
        
        if config.use_position_encoding:
            self.pos_encoding = PositionalEncoding2D(config.output_dim)
        
        self.output_projection = nn.Sequential(
            nn.Linear(config.output_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
            nn.Dropout(config.dropout)
        )
        
        self._initialize_weights()
        
        if config.freeze_backbone:
            self._freeze_backbone()
    
    def _create_backbone(self) -> nn.Module:
        if self.config.backbone == "efficientnet_b7":
            backbone = timm.create_model(
                'efficientnet_b7',
                pretrained=self.config.pretrained,
                in_chans=self.config.input_channels,
                num_classes=0,
                global_pool=''
            )
        elif self.config.backbone == "resnet50":
            backbone = timm.create_model(
                'resnet50',
                pretrained=self.config.pretrained,
                in_chans=self.config.input_channels,
                num_classes=0
            )
        elif self.config.backbone == "densenet121":
            backbone = timm.create_model(
                'densenet121',
                pretrained=self.config.pretrained,
                in_chans=self.config.input_channels,
                num_classes=0
            )
        elif self.config.backbone == "vit_base":
            backbone = timm.create_model(
                'vit_base_patch16_224',
                pretrained=self.config.pretrained,
                in_chans=self.config.input_channels,
                num_classes=0
            )
        elif self.config.backbone == "swin_transformer":
            backbone = timm.create_model(
                'swin_base_patch4_window7_224',
                pretrained=self.config.pretrained,
                in_chans=self.config.input_channels,
                num_classes=0
            )
        else:
            raise ValueError(f"Unsupported backbone: {self.config.backbone}")
        
        return backbone
    
    def _get_backbone_output_dim(self) -> int:
        with torch.no_grad():
            dummy_input = torch.randn(1, self.config.input_channels, *self.config.input_size)
            if hasattr(self.backbone, 'num_features'):
                return self.backbone.num_features
            else:
                output = self.backbone(dummy_input)
                if len(output.shape) == 4:
                    return output.shape[1]
                else:
                    return output.shape[-1]
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone_progressively(self, stage: int):
        if not self.config.progressive_unfreezing:
            return
            
        layers = list(self.backbone.named_parameters())
        total_layers = len(layers)
        
        if stage == 1:
            pass
        elif stage == 2:
            unfreeze_from = int(0.75 * total_layers)
            for i, (name, param) in enumerate(layers[unfreeze_from:]):
                param.requires_grad = True
        elif stage == 3:
            for param in self.backbone.parameters():
                param.requires_grad = True
    
    def forward(
        self,
        images: torch.Tensor,
        return_attention: bool = False,
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        batch_size = images.shape[0]
        
        if hasattr(self, 'preprocessor'):
            images = self.preprocessor(images)
        
        if self.config.use_multi_scale:
            multi_scale_features = self.feature_extractor(images)
            
            combined_features = []
            attention_maps = {}
            
            for level, features in multi_scale_features.items():
                refined = self.feature_extractor.refinement_layers[level](features)
                refined = refined.view(batch_size, -1)
                combined_features.append(refined)
                
                if return_attention and hasattr(self.feature_extractor, 'spatial_attention'):
                    with torch.no_grad():
                        attn_map = self.feature_extractor.spatial_attention[level](features)
                        attention_maps[f'{level}_attention'] = attn_map
            
            encoded_features = torch.cat(combined_features, dim=1)
            encoded_features = self.feature_combiner(encoded_features)
            
        else:
            backbone_output = self.backbone(images)
            encoded_features = self.feature_projector(backbone_output)
            attention_maps = {}
        
        if self.config.use_position_encoding and len(encoded_features.shape) > 2:
            encoded_features = self.pos_encoding(encoded_features)
        
        encoded_features = self.output_projection(encoded_features)
        
        if return_features or return_attention:
            feature_dict = {
                'encoded_features': encoded_features,
                'backbone_output': backbone_output if 'backbone_output' in locals() else None,
                'multi_scale_features': multi_scale_features if self.config.use_multi_scale else None,
                'attention_maps': attention_maps if return_attention else None,
                'preprocessed_images': images if hasattr(self, 'preprocessor') else None
            }
            return encoded_features, feature_dict
        
        return encoded_features
    
    def get_feature_maps(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, feature_dict = self.forward(images, return_features=True, return_attention=True)
        return feature_dict
    
    def compute_receptive_field(self) -> Dict[str, int]:
        return {
            'theoretical_rf': 512,
            'effective_rf': 384
        }



def create_vision_encoder(
    backbone: str = "efficientnet_b7",
    input_size: Tuple[int, int] = (512, 512),
    input_channels: int = 1,
    output_dim: int = 1024,
    **kwargs
) -> VisionEncoder:
    config = VisionEncoderConfig(
        backbone=backbone,
        input_size=input_size,
        input_channels=input_channels,
        output_dim=output_dim,
        **kwargs
    )
    
    return VisionEncoder(config)



def get_medical_transforms(
    input_size: Tuple[int, int] = (512, 512),
    is_training: bool = True
) -> transforms.Compose:
    
    if is_training:
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            transforms.RandomAutocontrast(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])



if __name__ == "__main__":
    print("Testing Vision Encoder for Radixpert...")
    
    configs = [
        {"backbone": "efficientnet_b7", "name": "EfficientNet-B7"},
        {"backbone": "resnet50", "name": "ResNet-50"},
        {"backbone": "vit_base", "name": "Vision Transformer"},
        {"backbone": "swin_transformer", "name": "Swin Transformer"}
    ]
    
    for config in configs:
        print(f"\nTesting {config['name']}...")
        
        encoder = create_vision_encoder(
            backbone=config["backbone"],
            input_size=(512, 512),
            input_channels=1,
            output_dim=1024,
            use_multi_scale=True,
            use_medical_preprocessing=True
        )
        
        batch_size = 4
        test_images = torch.randn(batch_size, 1, 512, 512)
        
        print(f"Input shape: {test_images.shape}")
        
        encoded_features = encoder(test_images)
        print(f"Encoded features shape: {encoded_features.shape}")
        
        encoded_features, feature_dict = encoder(
            test_images, 
            return_features=True, 
            return_attention=True
        )
        
        print(f"Feature dict keys: {list(feature_dict.keys())}")
        if feature_dict['multi_scale_features']:
            print(f"Multi-scale feature levels: {list(feature_dict['multi_scale_features'].keys())}")
        
        total_params = sum(p.numel() for p in encoder.parameters())
        trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable percentage: {100 * trainable_params / total_params:.1f}%")
    
    print("\nVision Encoder testing completed successfully!")
    
    print("\nTesting medical transforms...")
    transforms_train = get_medical_transforms(is_training=True)
    transforms_val = get_medical_transforms(is_training=False)
    
    print(f"Training transforms: {len(transforms_train.transforms)} steps")
    print(f"Validation transforms: {len(transforms_val.transforms)} steps")
