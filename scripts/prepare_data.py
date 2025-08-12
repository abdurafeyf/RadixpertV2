#!/usr/bin/env python3

import sys
import argparse
import logging
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import shutil
from collections import Counter
import re
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from PIL import Image
import pydicom


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from data.preprocessing import TextProcessor, MedicalImageProcessor
from utils.logging_utils import setup_comprehensive_logging


class DataPreparationPipeline:
    
    def __init__(self, output_dir: str, log_level: str = "INFO"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_comprehensive_logging(
            output_dir=str(self.output_dir),
            log_level=log_level,
            experiment_name="data_preparation"
        )
        
        try:
            from data.preprocessing import TextProcessor, MedicalImageProcessor
            class DummyConfig:
                max_text_length = 512
                image_size = (512, 512)
                
            config = DummyConfig()
            self.text_processor = TextProcessor(config)
            self.image_processor = MedicalImageProcessor(config)
        except ImportError:
            self.logger.warning("Could not import preprocessors - using basic preprocessing")
            self.text_processor = None
            self.image_processor = None
    
    def prepare_padchest_dataset(
        self,
        data_root: str,
        validate_images: bool = True,
        min_report_length: int = 10,
        max_report_length: int = 500
    ) -> Dict[str, Any]:
        self.logger.info("Starting Padchest dataset preparation...")
        
        data_root = Path(data_root)
        
        csv_files = list(data_root.glob("*PADCHEST*.csv")) + list(data_root.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in {data_root}")
        
        csv_file = csv_files[0]
        self.logger.info(f"Loading data from {csv_file}")
        
        df = pd.read_csv(csv_file)
        self.logger.info(f"Loaded {len(df)} samples from Padchest CSV")
        
        original_count = len(df)
        
        df = df.dropna(subset=['Report'])
        df = df[df['Report'].str.strip() != '']
        self.logger.info(f"After removing empty reports: {len(df)} samples")
        
        df['report_word_count'] = df['Report'].str.split().str.len()
        df = df[
            (df['report_word_count'] >= min_report_length) & 
            (df['report_word_count'] <= max_report_length)
        ]
        self.logger.info(f"After report length filtering: {len(df)} samples")
        
        self.logger.info("Cleaning and normalizing reports...")
        df['cleaned_report'] = df['Report'].apply(self._clean_spanish_report)
        df['translated_report'] = df['cleaned_report'].apply(self._translate_spanish_keywords)
        
        if validate_images:
            self.logger.info("Validating image files...")
            valid_mask = df.apply(lambda row: self._validate_padchest_image(data_root, row), axis=1)
            df = df[valid_mask]
            self.logger.info(f"After image validation: {len(df)} samples")
        
        df_splits = self._create_dataset_splits(df, seed=42)
        
        processed_dir = self.output_dir / "padchest_processed"
        processed_dir.mkdir(exist_ok=True)
        
        for split, split_df in df_splits.items():
            split_file = processed_dir / f"{split}.csv"
            split_df.to_csv(split_file, index=False)
            self.logger.info(f"Saved {split} split ({len(split_df)} samples) to {split_file}")
        
        stats = self._generate_padchest_statistics(df, df_splits)
        
        stats_file = processed_dir / "dataset_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info("Padchest dataset preparation complete")
        return stats
    
    def prepare_roco_dataset(
        self,
        data_root: str,
        validate_images: bool = True,
        min_caption_length: int = 5,
        max_caption_length: int = 200
    ) -> Dict[str, Any]:
        self.logger.info("Starting ROCO v2 dataset preparation...")
        
        data_root = Path(data_root)
        
        annotation_files = list(data_root.glob("*.json")) + list(data_root.glob("*.csv"))
        if not annotation_files:
            raise FileNotFoundError(f"No annotation file found in {data_root}")
        
        annotation_file = annotation_files[0]
        self.logger.info(f"Loading data from {annotation_file}")
        
        if annotation_file.suffix == '.json':
            with open(annotation_file, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.json_normalize(data)
        else:
            df = pd.read_csv(annotation_file)
        
        self.logger.info(f"Loaded {len(df)} samples from ROCO annotations")
        
        original_count = len(df)
        
        caption_col = None
        for col in ['caption', 'text', 'description', 'Caption', 'Text']:
            if col in df.columns:
                caption_col = col
                break
        if caption_col is None:
            raise ValueError("No caption column found in ROCO data")
        
        df = df.dropna(subset=[caption_col])
        df = df[df[caption_col].str.strip() != '']
        self.logger.info(f"After removing empty captions: {len(df)} samples")
        
        df['caption_word_count'] = df[caption_col].str.split().str.len()
        df = df[
            (df['caption_word_count'] >= min_caption_length) &
            (df['caption_word_count'] <= max_caption_length)
        ]
        self.logger.info(f"After caption length filtering: {len(df)} samples")
        
        self.logger.info("Cleaning and normalizing captions...")
        df['cleaned_caption'] = df[caption_col].apply(self._clean_english_caption)
        
        df['is_medical'] = df['cleaned_caption'].apply(self._is_medical_content)
        df = df[df['is_medical']]
        self.logger.info(f"After medical content filtering: {len(df)} samples")
        
        if validate_images:
            self.logger.info("Validating image files...")
            valid_mask = df.apply(lambda row: self._validate_roco_image(data_root, row), axis=1)
            df = df[valid_mask]
            self.logger.info(f"After image validation: {len(df)} samples")
        
        df_splits = self._create_dataset_splits(df, seed=42)
        
        processed_dir = self.output_dir / "roco_processed"
        processed_dir.mkdir(exist_ok=True)
        
        for split, split_df in df_splits.items():
            split_file = processed_dir / f"{split}.csv"
            split_df.to_csv(split_file, index=False)
            self.logger.info(f"Saved {split} split ({len(split_df)} samples) to {split_file}")
        
        stats = self._generate_roco_statistics(df, df_splits)
        
        stats_file = processed_dir / "dataset_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info("ROCO v2 dataset preparation complete")
        return stats
    
    # Helper methods (_clean_spanish_report, _translate_spanish_keywords,
    # _clean_english_caption, _is_medical_content, _validate_padchest_image,
    # _validate_roco_image, _create_dataset_splits, _generate_padchest_statistics,
    # _generate_roco_statistics) omitted here for brevity - as per original code.
    
    # Other methods for processing metadata, creating unified annotations, etc., follow accordingly.
    
    
def main():
    args = parse_arguments()
    
    if args.dataset == "both":
        if not args.padchest_root or not args.roco_root:
            print("Error: --padchest_root and --roco_root are required for 'both'")
            return 1
    else:
        if not args.data_root:
            print("Error: --data_root is required")
            return 1
    
    pipeline = DataPreparationPipeline(output_dir=args.output_dir, log_level=args.log_level)
    pipeline.logger.info("=" * 80)
    pipeline.logger.info("RADIXPERT DATA PREPARATION STARTED")
    pipeline.logger.info("=" * 80)
    
    try:
        all_stats = {}
        padchest_samples = []
        roco_samples = []
        
        if args.dataset in ("padchest", "both"):
            padchest_dir = args.padchest_root if args.dataset == "both" else args.data_root
            stats = pipeline.prepare_padchest_dataset(
                data_root=padchest_dir,
                validate_images=args.validate_images,
                min_report_length=args.min_report_length,
                max_report_length=args.max_report_length,
            )
            all_stats['padchest'] = stats
            padchest_samples = pipeline.process_padchest_metadata(padchest_dir, Path(args.output_dir), pipeline.logger)
        
        if args.dataset in ("roco", "both"):
            roco_dir = args.roco_root if args.dataset == "both" else args.data_root
            stats = pipeline.prepare_roco_dataset(
                data_root=roco_dir,
                validate_images=args.validate_images,
                min_caption_length=args.min_caption_length,
                max_caption_length=args.max_caption_length,
            )
            all_stats['roco'] = stats
            roco_samples = pipeline.process_roco_metadata(roco_dir, Path(args.output_dir), pipeline.logger)
        
        if padchest_samples and roco_samples:
            pipeline.create_unified_annotations(padchest_samples, roco_samples, Path(args.output_dir), pipeline.logger)
        
        overall_stats = {
            'total_datasets': len(all_stats),
            'datasets': list(all_stats.keys()),
            'total_samples': sum(s.get('total_samples', 0) for s in all_stats.values()),
            'processing_args': vars(args),
            'dataset_stats': all_stats,
        }
        
        stats_file = Path(args.output_dir) / "overall_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(overall_stats, f, indent=2)
        
        pipeline.logger.info("Data preparation completed successfully")
        pipeline.logger.info(f"Data saved to {args.output_dir}")
        return 0
    
    except Exception as e:
        pipeline.logger.error(f"Data preparation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
