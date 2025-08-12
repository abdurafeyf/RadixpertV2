import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pydicom
from sklearn.model_selection import train_test_split
import re


from preprocessing import MedicalImagePreprocessor, TextPreprocessor
from transforms import get_medical_transforms


@dataclass
class DatasetConfig:
    data_root: str
    images_dir: str
    annotations_file: str
    
    image_size: Tuple[int, int] = (512, 512)
    max_text_length: int = 512
    min_text_length: int = 10
    
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    modality: str = "chest_xray"
    view_position: Optional[str] = None
    
    use_medical_preprocessing: bool = True
    use_data_augmentation: bool = True
    normalize_text: bool = True
    
    filter_by_quality: bool = True
    min_image_resolution: Tuple[int, int] = (256, 256)
    exclude_lateral_views: bool = False


class BaseRadiologyDataset(Dataset, ABC):
    
    def __init__(
        self,
        config: DatasetConfig,
        split: str = "train",
        stage: int = 1,
        transform_type: str = "train"
    ):
        self.config = config
        self.split = split
        self.stage = stage
        self.transform_type = transform_type
        
        self.data_root = Path(config.data_root)
        self.images_dir = self.data_root / config.images_dir
        self.annotations_file = self.data_root / config.annotations_file
        
        self.image_preprocessor = MedicalImagePreprocessor(config)
        self.text_preprocessor = TextPreprocessor(config)
        
        self.transforms = get_medical_transforms(
            input_size=config.image_size,
            is_training=(transform_type == "train"),
            use_medical_augmentation=config.use_data_augmentation
        )
        
        self.data = self._load_data()
        self.data = self._filter_data(self.data)
        self.data = self._create_splits(self.data)
        
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.info(f"Initialized {self.__class__.__name__} with {len(self.data)} samples")
    
    @abstractmethod
    def _load_data(self) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def _get_image_path(self, sample: Dict[str, Any]) -> str:
        pass
    
    @abstractmethod
    def _get_text_content(self, sample: Dict[str, Any]) -> str:
        pass
    
    def _filter_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered_data = []
        
        for sample in data:
            text = self._get_text_content(sample)
            if len(text.split()) < self.config.min_text_length:
                continue
            
            image_path = self._get_image_path(sample)
            if not os.path.exists(image_path):
                continue
            
            if self.config.filter_by_quality:
                if not self._passes_quality_check(sample):
                    continue
            
            filtered_data.append(sample)
        
        self.logger.info(f"Filtered data: {len(data)} -> {len(filtered_data)} samples")
        return filtered_data
    
    def _passes_quality_check(self, sample: Dict[str, Any]) -> bool:
        return True
    
    def _create_splits(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.split == "all":
            return data
        
        train_data, temp_data = train_test_split(
            data, 
            test_size=(1 - self.config.train_split),
            random_state=42,
            shuffle=True
        )
        
        val_data, test_data = train_test_split(
            temp_data,
            test_size=self.config.test_split / (1 - self.config.train_split),
            random_state=42,
            shuffle=True
        )
        
        if self.split == "train":
            return train_data
        elif self.split == "val":
            return val_data
        elif self.split == "test":
            return test_data
        else:
            raise ValueError(f"Unknown split: {self.split}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        
        image = self._load_image(sample)
        image = self.transforms(image)
        
        text = self._get_text_content(sample)
        text_data = self.text_preprocessor.process_text(text)
        
        return {
            'images': image,
            'input_ids': text_data['input_ids'],
            'attention_mask': text_data['attention_mask'],
            'labels': text_data['labels'],
            'text': text,
            'image_path': self._get_image_path(sample),
            'sample_id': sample.get('id', idx),
            'metadata': self._get_metadata(sample)
        }
    
    def _load_image(self, sample: Dict[str, Any]) -> Image.Image:
        image_path = self._get_image_path(sample)
        
        try:
            if image_path.lower().endswith('.dcm'):
                dicom_data = pydicom.dcmread(image_path)
                image_array = dicom_data.pixel_array
                
                image_array = ((image_array - image_array.min()) / 
                              (image_array.max() - image_array.min()) * 255).astype(np.uint8)
                
                image = Image.fromarray(image_array, mode='L')
            else:
                image = Image.open(image_path).convert('L')
            
            if self.config.use_medical_preprocessing:
                image = self.image_preprocessor.preprocess_image(image)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {e}")
            return Image.new('L', self.config.image_size, 0)
    
    def _get_metadata(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'dataset': self.__class__.__name__,
            'split': self.split,
            'stage': self.stage
        }


class PadChestDataset(BaseRadiologyDataset):
    
    def __init__(self, config: DatasetConfig, **kwargs):
        config.modality = "chest_xray"
        super().__init__(config, **kwargs)
    
    def _load_data(self) -> List[Dict[str, Any]]:
        df = pd.read_csv(self.annotations_file)
        
        data = []
        for idx, row in df.iterrows():
            if pd.isna(row.get('Report', '')) or row.get('Report', '') == '':
                continue
            
            sample = {
                'id': row.get('ImageID', f'padchest_{idx}'),
                'image_name': row.get('ImageID', ''),
                'report': row.get('Report', ''),
                'findings': row.get('Labels', ''),
                'view_position': row.get('Projection', ''),
                'patient_id': row.get('PatientID', ''),
                'study_id': row.get('StudyID', ''),
                'age': row.get('PatientAge', ''),
                'sex': row.get('PatientSex', ''),
                'modality': row.get('Modality', 'DX'),
                'original_row': row.to_dict()
            }
            data.append(sample)
        
        self.logger.info(f"Loaded {len(data)} PadChest samples")
        return data
    
    def _get_image_path(self, sample: Dict[str, Any]) -> str:
        image_name = sample['image_name']
        
        if len(image_name) >= 4:
            subdir = '/'.join(image_name[:4])
            image_path = self.images_dir / subdir / f"{image_name}.png"
        else:
            image_path = self.images_dir / f"{image_name}.png"
        
        return str(image_path)
    
    def _get_text_content(self, sample: Dict[str, Any]) -> str:
        report = sample.get('report', '')
        findings = sample.get('findings', '')
        
        if findings and findings != '':
            text = f"FINDINGS: {findings}. REPORT: {report}"
        else:
            text = f"REPORT: {report}"
        
        return text.strip()
    
    def _passes_quality_check(self, sample: Dict[str, Any]) -> bool:
        if self.config.view_position:
            view = sample.get('view_position', '').upper()
            if self.config.view_position.upper() not in view:
                return False
        
        if self.config.exclude_lateral_views:
            view = sample.get('view_position', '').upper()
            if 'LAT' in view or 'LATERAL' in view:
                return False
        
        report = sample.get('report', '')
        if len(report.split()) < 5:
            return False
        
        return True
    
    def _get_metadata(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        base_metadata = super()._get_metadata(sample)
        base_metadata.update({
            'patient_id': sample.get('patient_id', ''),
            'study_id': sample.get('study_id', ''),
            'view_position': sample.get('view_position', ''),
            'age': sample.get('age', ''),
            'sex': sample.get('sex', ''),
            'language': 'spanish'
        })
        return base_metadata


class ROCOv2Dataset(BaseRadiologyDataset):
    
    def __init__(self, config: DatasetConfig, **kwargs):
        config.modality = "mixed_radiology"
        super().__init__(config, **kwargs)
    
    def _load_data(self) -> List[Dict[str, Any]]:
        if self.annotations_file.endswith('.json'):
            with open(self.annotations_file, 'r') as f:
                raw_data = json.load(f)
        else:
            df = pd.read_csv(self.annotations_file)
            raw_data = df.to_dict('records')
        
        data = []
        for idx, item in enumerate(raw_data):
            if isinstance(item, dict):
                sample = {
                    'id': item.get('id', f'roco_{idx}'),
                    'image_name': item.get('name', item.get('image', '')),
                    'caption': item.get('caption', item.get('text', '')),
                    'keywords': item.get('keywords', ''),
                    'modality': item.get('modality', 'radiography'),
                    'anatomy': item.get('anatomy', ''),
                    'original_data': item
                }
            else:
                sample = {
                    'id': f'roco_{idx}',
                    'image_name': str(item),
                    'caption': '',
                    'keywords': '',
                    'modality': 'radiography',
                    'anatomy': '',
                    'original_data': item
                }
            
            if sample['caption'] or sample['keywords']:
                data.append(sample)
        
        self.logger.info(f"Loaded {len(data)} ROCO v2 samples")
        return data
    
    def _get_image_path(self, sample: Dict[str, Any]) -> str:
        image_name = sample['image_name']
        
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            if image_name.lower().endswith(ext):
                image_path = self.images_dir / image_name
                break
        else:
            image_path = self.images_dir / f"{image_name}.jpg"
        
        return str(image_path)
    
    def _get_text_content(self, sample: Dict[str, Any]) -> str:
        caption = sample.get('caption', '')
        keywords = sample.get('keywords', '')
        anatomy = sample.get('anatomy', '')
        
        text_parts = []
        
        if anatomy:
            text_parts.append(f"ANATOMY: {anatomy}")
        
        if keywords:
            text_parts.append(f"KEYWORDS: {keywords}")
        
        if caption:
            text_parts.append(f"CAPTION: {caption}")
        
        return '. '.join(text_parts).strip()
    
    def _passes_quality_check(self, sample: Dict[str, Any]) -> bool:
        caption = sample.get('caption', '')
        if len(caption.split()) < 3:
            return False
        
        medical_keywords = [
            'radiograph', 'ct', 'mri', 'ultrasound', 'x-ray', 'scan',
            'patient', 'diagnosis', 'findings', 'anatomy', 'pathology',
            'chest', 'abdomen', 'brain', 'spine', 'bone', 'tissue'
        ]
        
        text_lower = (caption + ' ' + sample.get('keywords', '')).lower()
        if not any(keyword in text_lower for keyword in medical_keywords):
            return False
        
        return True
    
    def _get_metadata(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        base_metadata = super()._get_metadata(sample)
        base_metadata.update({
            'modality': sample.get('modality', ''),
            'anatomy': sample.get('anatomy', ''),
            'language': 'english',
            'source': 'medical_literature'
        })
        return base_metadata


class MultiDatasetLoader:
    
    def __init__(
        self,
        padchest_config: DatasetConfig,
        roco_config: DatasetConfig,
        stage: int = 1,
        batch_size: int = 16,
        num_workers: int = 4
    ):
        self.padchest_config = padchest_config
        self.roco_config = roco_config
        self.stage = stage
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.logger = logging.getLogger("MultiDatasetLoader")
        
        self.datasets = self._create_datasets_for_stage()
    
    def _create_datasets_for_stage(self) -> Dict[str, Dict[str, Dataset]]:
        datasets = {}
        
        if self.stage == 1:
            self.logger.info("Stage 1: Loading ROCO v2 only")
            datasets['roco_v2'] = {
                'train': ROCOv2Dataset(self.roco_config, split='train', stage=1),
                'val': ROCOv2Dataset(self.roco_config, split='val', stage=1),
                'test': ROCOv2Dataset(self.roco_config, split='test', stage=1)
            }
            
        elif self.stage == 2:
            self.logger.info("Stage 2: Loading ROCO v2 + PadChest")
            datasets['roco_v2'] = {
                'train': ROCOv2Dataset(self.roco_config, split='train', stage=2),
                'val': ROCOv2Dataset(self.roco_config, split='val', stage=2),
                'test': ROCOv2Dataset(self.roco_config, split='test', stage=2)
            }
            datasets['padchest'] = {
                'train': PadChestDataset(self.padchest_config, split='train', stage=2),
                'val': PadChestDataset(self.padchest_config, split='val', stage=2),
                'test': PadChestDataset(self.padchest_config, split='test', stage=2)
            }
            
        elif self.stage == 3:
            self.logger.info("Stage 3: Loading combined datasets")
            datasets['roco_v2'] = {
                'train': ROCOv2Dataset(self.roco_config, split='train', stage=3),
                'val': ROCOv2Dataset(self.roco_config, split='val', stage=3),
                'test': ROCOv2Dataset(self.roco_config, split='test', stage=3)
            }
            datasets['padchest'] = {
                'train': PadChestDataset(self.padchest_config, split='train', stage=3),
                'val': PadChestDataset(self.padchest_config, split='val', stage=3),
                'test': PadChestDataset(self.padchest_config, split='test', stage=3)
            }
        
        return datasets
    
    def get_dataloaders(self, split: str = 'train') -> Dict[str, DataLoader]:
        dataloaders = {}
        
        for dataset_name, splits in self.datasets.items():
            if split in splits:
                dataset = splits[split]
                
                dataloader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=(split == 'train'),
                    num_workers=self.num_workers,
                    pin_memory=True,
                    drop_last=(split == 'train'),
                    collate_fn=self._collate_fn
                )
                
                dataloaders[dataset_name] = dataloader
                
                self.logger.info(
                    f"Created {dataset_name} {split} loader: "
                    f"{len(dataset)} samples, {len(dataloader)} batches"
                )
        
        return dataloaders
    
    def get_combined_dataloader(self, split: str = 'train') -> DataLoader:
        datasets_list = []
        
        for dataset_name, splits in self.datasets.items():
            if split in splits:
                datasets_list.append(splits[split])
        
        if not datasets_list:
            raise ValueError(f"No datasets available for split: {split}")
        
        combined_dataset = torch.utils.data.ConcatDataset(datasets_list)
        
        combined_dataloader = DataLoader(
            combined_dataset,
            batch_size=self.batch_size,
            shuffle=(split == 'train'),
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=(split == 'train'),
            collate_fn=self._collate_fn
        )
        
        self.logger.info(
            f"Created combined {split} loader: "
            f"{len(combined_dataset)} samples, {len(combined_dataloader)} batches"
        )
        
        return combined_dataloader
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = torch.stack([item['images'] for item in batch])
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        texts = [item['text'] for item in batch]
        image_paths = [item['image_path'] for item in batch]
        sample_ids = [item['sample_id'] for item in batch]
        metadata = [item['metadata'] for item in batch]
        
        return {
            'images': images,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'texts': texts,
            'image_paths': image_paths,
            'sample_ids': sample_ids,
            'metadata': metadata
        }
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        stats = {
            'stage': self.stage,
            'datasets': {},
            'total_samples': 0
        }
        
        for dataset_name, splits in self.datasets.items():
            dataset_stats = {
                'splits': {},
                'total': 0
            }
            
            for split_name, dataset in splits.items():
                split_size = len(dataset)
                dataset_stats['splits'][split_name] = split_size
                dataset_stats['total'] += split_size
            
            stats['datasets'][dataset_name] = dataset_stats
            stats['total_samples'] += dataset_stats['total']
        
        return stats


def create_padchest_config(
    data_root: str,
    images_dir: str = "images-64",
    annotations_file: str = "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv",
    **kwargs
) -> DatasetConfig:
    return DatasetConfig(
        data_root=data_root,
        images_dir=images_dir,
        annotations_file=annotations_file,
        **kwargs
    )


def create_roco_config(
    data_root: str,
    images_dir: str = "images",
    annotations_file: str = "roco-dataset.json",
    **kwargs
) -> DatasetConfig:
    return DatasetConfig(
        data_root=data_root,
        images_dir=images_dir,
        annotations_file=annotations_file,
        **kwargs
    )


def create_multi_stage_datasets(
    padchest_root: str,
    roco_root: str,
    batch_size: int = 16,
    num_workers: int = 4
) -> Dict[int, MultiDatasetLoader]:
    padchest_config = create_padchest_config(padchest_root)
    roco_config = create_roco_config(roco_root)
    
    stage_loaders = {}
    
    for stage in [1, 2, 3]:
        stage_loaders[stage] = MultiDatasetLoader(
            padchest_config=padchest_config,
            roco_config=roco_config,
            stage=stage,
            batch_size=batch_size,
            num_workers=num_workers
        )
    
    return stage_loaders


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    padchest_root = "/path/to/padchest"
    roco_root = "/path/to/roco_v2"
    
    print("Creating multi-stage datasets...")
    
    stage_datasets = create_multi_stage_datasets(
        padchest_root=padchest_root,
        roco_root=roco_root,
        batch_size=8,
        num_workers=4
    )
    
    for stage_num, loader in stage_datasets.items():
        print(f"\n=== Stage {stage_num} ===")
        
        stats = loader.get_dataset_statistics()
        print(f"Statistics: {stats}")
        
        train_loaders = loader.get_dataloaders('train')
        print(f"Available train loaders: {list(train_loaders.keys())}")
        
        for dataset_name, dataloader in train_loaders.items():
            print(f"\nTesting {dataset_name}...")
            batch = next(iter(dataloader))
            
            print(f"Batch keys: {list(batch.keys())}")
            print(f"Images shape: {batch['images'].shape}")
            print(f"Input IDs shape: {batch['input_ids'].shape}")
            print(f"First text sample: {batch['texts'][0][:100]}...")
            break
    
    print("Dataset testing completed successfully!")
