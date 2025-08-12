import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import random
from typing import Tuple, Optional


class MedicalAugmentation:
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))
            
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.9, 1.1))
            
            if random.random() < 0.3:
                angle = random.uniform(-3, 3)
                img = img.rotate(angle, fillcolor=0)
        
        return img


class NoiseReduction:
    
    def __init__(self, p: float = 0.3):
        self.p = p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        return img


def get_medical_transforms(
    input_size: Tuple[int, int] = (512, 512),
    is_training: bool = True,
    use_medical_augmentation: bool = True
) -> transforms.Compose:
    transform_list = []
    
    transform_list.append(transforms.Resize(input_size))
    
    if is_training:
        transform_list.extend([
            transforms.RandomRotation(5, fill=0),
            transforms.RandomHorizontalFlip(p=0.2),
        ])
        
        if use_medical_augmentation:
            transform_list.extend([
                MedicalAugmentation(p=0.4),
                NoiseReduction(p=0.3)
            ])
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    return transforms.Compose(transform_list)
