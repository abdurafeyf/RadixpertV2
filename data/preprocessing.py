import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Dict, List, Optional, Tuple, Union
import cv2
from transformers import AutoTokenizer
import re
from dataclasses import dataclass


@dataclass
class PreprocessingConfig:
    target_size: Tuple[int, int] = (512, 512)
    normalize_intensity: bool = True
    enhance_contrast: bool = True
    reduce_noise: bool = True
    
    max_text_length: int = 512
    tokenizer_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    add_special_tokens: bool = True
    
    window_center: float = 0.5
    window_width: float = 1.0
    apply_clahe: bool = True


class MedicalImagePreprocessor:
    
    def __init__(self, config: Union[PreprocessingConfig, DatasetConfig]):
        if hasattr(config, 'image_size'):
            self.config = PreprocessingConfig(
                target_size=config.image_size,
                max_text_length=config.max_text_length
            )
        else:
            self.config = config
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        image = image.resize(self.config.target_size, Image.Resampling.LANCZOS)
        
        img_array = np.array(image)
        
        if self.config.normalize_intensity:
            img_array = self._normalize_intensity(img_array)
        
        if self.config.apply_clahe:
            img_array = self._apply_clahe(img_array)
        
        if self.config.enhance_contrast:
            img_array = self._enhance_contrast(img_array)
        
        if self.config.reduce_noise:
            img_array = self._reduce_noise(img_array)
        
        return Image.fromarray(img_array.astype(np.uint8), mode='L')
    
    def _normalize_intensity(self, img_array: np.ndarray) -> np.ndarray:
        img_min, img_max = img_array.min(), img_array.max()
        if img_max > img_min:
            img_array = ((img_array - img_min) / (img_max - img_min) * 255)
        return img_array
    
    def _apply_clahe(self, img_array: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img_array.astype(np.uint8))
    
    def _enhance_contrast(self, img_array: np.ndarray) -> np.ndarray:
        gamma = 1.2
        img_array = np.power(img_array / 255.0, gamma) * 255.0
        return np.clip(img_array, 0, 255)
    
    def _reduce_noise(self, img_array: np.ndarray) -> np.ndarray:
        return cv2.bilateralFilter(
            img_array.astype(np.uint8), 
            d=5, 
            sigmaColor=75, 
            sigmaSpace=75
        )


class TextPreprocessor:
    
    def __init__(self, config: Union[PreprocessingConfig, DatasetConfig]):
        if hasattr(config, 'max_text_length'):
            self.max_length = config.max_text_length
            tokenizer_name = getattr(config, 'tokenizer_name', "meta-llama/Llama-3.2-11B-Vision-Instruct")
        else:
            self.max_length = config.max_text_length
            tokenizer_name = config.tokenizer_name
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def process_text(self, text: str) -> Dict[str, torch.Tensor]:
        text = self._clean_text(text)
        
        text = self._add_medical_structure(text)
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = encoding.input_ids.clone()
        
        return {
            'input_ids': encoding.input_ids.squeeze(0),
            'attention_mask': encoding.attention_mask.squeeze(0),
            'labels': labels.squeeze(0)
        }
    
    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        
        text = self._normalize_abbreviations(text)
        
        if self._is_spanish(text):
            text = self._translate_spanish_keywords(text)
        
        return text
    
    def _normalize_abbreviations(self, text: str) -> str:
        abbreviations = {
            'PA': 'posteroanterior',
            'AP': 'anteroposterior', 
            'LAT': 'lateral',
            'CXR': 'chest X-ray',
            'CT': 'computed tomography',
            'MRI': 'magnetic resonance imaging'
        }
        
        for abbr, full in abbreviations.items():
            text = re.sub(r'\b' + abbr + r'\b', full, text, flags=re.IGNORECASE)
        
        return text
    
    def _is_spanish(self, text: str) -> bool:
        spanish_indicators = ['radiografía', 'tórax', 'pulmón', 'corazón', 'paciente']
        return any(indicator in text.lower() for indicator in spanish_indicators)
    
    def _translate_spanish_keywords(self, text: str) -> str:
        translations = {
            'radiografía': 'radiograph',
            'tórax': 'chest',
            'pulmón': 'lung',
            'pulmones': 'lungs',
            'corazón': 'heart',
            'paciente': 'patient',
            'normal': 'normal',
            'anormal': 'abnormal',
            'izquierdo': 'left',
            'derecho': 'right'
        }
        
        for spanish, english in translations.items():
            text = text.replace(spanish, english)
        
        return text
    
    def _add_medical_structure(self, text: str) -> str:
        if not text.startswith(('FINDINGS:', 'IMPRESSION:', 'REPORT:')):
            text = f"FINDINGS: {text}"
        
        return text
