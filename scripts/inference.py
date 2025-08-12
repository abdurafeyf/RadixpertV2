#!/usr/bin/env python3

import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from PIL import Image
from ..models.radixpert import Radixpert, RadixpertConfig
from ..data.transforms import get_medical_transforms
from ..utils.device_utils import setup_device
from ..utils.checkpoint_utils import CheckpointManager
from ..utils.logging_utils import setup_comprehensive_logging


class RadixpertInference:
    
    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None,
        model_config: Optional[Dict] = None
    ):
        self.model_path = model_path
        self.device = device or setup_device()
        self.model = self._load_model(model_config)
        self.transforms = get_medical_transforms(
            input_size=(512, 512),
            is_training=False
        )
        self.logger = logging.getLogger("RadixpertInference")
        self.logger.info("Radixpert inference engine initialized")
    
    def _load_model(self, model_config: Optional[Dict] = None) -> Radixpert:
        if model_config:
            config = RadixpertConfig(**model_config)
        else:
            config = RadixpertConfig()
        model = Radixpert(config)
        model = model.to(self.device)
        checkpoint_manager = CheckpointManager(Path(self.model_path).parent)
        checkpoint_manager.load_checkpoint(self.model_path, model, device=self.device)
        model.eval()
        return model
    
    def generate_report(
        self,
        image: Union[str, Path, Image.Image],
        prompt: Optional[str] = None,
        max_length: int = 512,
        temperature: float = 0.7,
        num_beams: int = 4,
        return_confidence: bool = False
    ) -> Dict[str, Union[str, float]]:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('L')
        image_tensor = self.transforms(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            start_time = time.time()
            if prompt:
                reports = self.model.generate_report(
                    images=image_tensor,
                    prompt=prompt,
                    max_length=max_length,
                    temperature=temperature,
                    num_beams=num_beams
                )
            else:
                reports = self.model.generate_report(
                    images=image_tensor,
                    max_length=max_length,
                    temperature=temperature,
                    num_beams=num_beams
                )
            generation_time = time.time() - start_time
        result = {
            'report': reports[0] if reports else "",
            'generation_time_seconds': generation_time,
            'model_path': str(self.model_path),
            'parameters': {
                'max_length': max_length,
                'temperature': temperature,
                'num_beams': num_beams
            }
        }
        if return_confidence:
            confidence = self._estimate_confidence(image_tensor, reports[0])
            result['confidence_score'] = confidence
        return result
    
    def batch_inference(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 8,
        **generation_kwargs
    ) -> List[Dict[str, Union[str, float]]]:
        results = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(image_paths)-1)//batch_size + 1}")
            batch_images = []
            valid_paths = []
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('L')
                    image_tensor = self.transforms(image)
                    batch_images.append(image_tensor)
                    valid_paths.append(path)
                except Exception as e:
                    self.logger.warning(f"Failed to load image {path}: {e}")
                    continue
            if not batch_images:
                continue
            batch_tensor = torch.stack(batch_images).to(self.device)
            with torch.no_grad():
                start_time = time.time()
                batch_reports = self.model.generate_report(
                    images=batch_tensor,
                    **generation_kwargs
                )
                generation_time = time.time() - start_time
            for path, report in zip(valid_paths, batch_reports):
                result = {
                    'image_path': str(path),
                    'report': report,
                    'generation_time_seconds': generation_time / len(batch_reports),
                    'model_path': str(self.model_path),
                    'parameters': generation_kwargs
                }
                results.append(result)
        return results
    
    def _estimate_confidence(self, image_tensor: torch.Tensor, report: str) -> float:
        confidence = 0.0
        words = report.split()
        word_count = len(words)
        if 20 <= word_count <= 200:
            length_score = 1.0
        elif word_count < 20:
            length_score = word_count / 20.0
        else:
            length_score = max(0.0, 1.0 - (word_count - 200) / 100.0)
        confidence += length_score * 0.3
        medical_terms = [
            'lung', 'heart', 'chest', 'radiograph', 'findings', 'impression',
            'normal', 'abnormal', 'opacity', 'consolidation', 'pleural'
        ]
        report_lower = report.lower()
        term_score = sum(1 for term in medical_terms if term in report_lower) / len(medical_terms)
        confidence += term_score * 0.4
        structure_indicators = ['findings:', 'impression:', 'technique:']
        structure_score = min(1.0, sum(1 for ind in structure_indicators if ind in report_lower) * 0.5)
        confidence += structure_score * 0.3
        return min(1.0, confidence)
    
    def format_clinical_report(self, report: str, patient_info: Optional[Dict] = None) -> str:
        formatted_report = []
        formatted_report.append("RADIOLOGY REPORT")
        formatted_report.append("=" * 50)
        formatted_report.append("")
        if patient_info:
            formatted_report.append("PATIENT INFORMATION:")
            for key, value in patient_info.items():
                formatted_report.append(f"  {key.title()}: {value}")
            formatted_report.append("")
        formatted_report.append("GENERATED REPORT:")
        formatted_report.append("-" * 30)
        sections = self._parse_report_sections(report)
        for section, content in sections.items():
            if content.strip():
                formatted_report.append(f"{section.upper()}:")
                formatted_report.append(f"  {content.strip()}")
                formatted_report.append("")
        formatted_report.append("-" * 50)
        formatted_report.append("Generated by Radixpert AI System")
        formatted_report.append(f"Model: {Path(self.model_path).name}")
        formatted_report.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return "\n".join(formatted_report)
    
    def _parse_report_sections(self, report: str) -> Dict[str, str]:
        sections = {
            'technique': '',
            'findings': '',
            'impression': ''
        }
        report_lower = report.lower()
        if 'technique:' in report_lower:
            start = report_lower.find('technique:') + len('technique:')
            end = min([report_lower.find(section + ':', start) for section in ['findings', 'impression'] if report_lower.find(section + ':', start) != -1] + [len(report)])
            sections['technique'] = report[start:end].strip()
        if 'findings:' in report_lower:
            start = report_lower.find('findings:') + len('findings:')
            end = report_lower.find('impression:', start) if 'impression:' in report_lower[start:] else len(report)
            sections['findings'] = report[start:end].strip()
        if 'impression:' in report_lower:
            start = report_lower.find('impression:') + len('impression:')
            sections['impression'] = report[start:].strip()
        if not any(sections.values()):
            sections['findings'] = report.strip()
        return sections


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate radiology reports using Radixpert",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained Radixpert model checkpoint")
    parser.add_argument("--model_config", type=str, default=None, help="Path to model configuration JSON file")
    parser.add_argument("--image_path", type=str, default=None, help="Path to single image for inference")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory containing images for batch processing")
    parser.add_argument("--batch_mode", action="store_true", help="Enable batch processing mode")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for batch processing")
    parser.add_argument("--prompt", type=str, default=None, help="Optional text prompt for guided generation")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length of generated report")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for generation")
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams for beam search")
    parser.add_argument("--output_dir", type=str, default="./inference_results", help="Directory to save results")
    parser.add_argument("--save_formatted", action="store_true", help="Save formatted clinical reports")
    parser.add_argument("--return_confidence", action="store_true", help="Include confidence scores in results")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"], help="Device to use for inference")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if not args.image_path and not args.image_dir:
        print("Error: Must specify either --image_path or --image_dir")
        return 1
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_comprehensive_logging(
        output_dir=args.output_dir,
        log_level=args.log_level,
        experiment_name="radixpert_inference"
    )
    logger.info("=" * 80)
    logger.info("RADIXPERT CLINICAL INFERENCE STARTED")
    logger.info("=" * 80)
    try:
        device = setup_device(logger) if args.device == "auto" else torch.device(args.device)
        model_config = None
        if args.model_config:
            with open(args.model_config, 'r') as f:
                model_config = json.load(f)
        inference_engine = RadixpertInference(
            model_path=args.model_path,
            device=device,
            model_config=model_config
        )
        generation_kwargs = {
            'prompt': args.prompt,
            'max_length': args.max_length,
            'temperature': args.temperature,
            'num_beams': args.num_beams,
            'return_confidence': args.return_confidence
        }
        if args.image_path and not args.batch_mode:
            logger.info(f"Processing single image: {args.image_path}")
            result = inference_engine.generate_report(
                image=args.image_path,
                **generation_kwargs
            )
            logger.info("Generated Report:")
            logger.info("-" * 50)
            logger.info(result['report'])
            logger.info("-" * 50)
            if args.return_confidence:
                logger.info(f"Confidence Score: {result['confidence_score']:.3f}")
            logger.info(f"Generation Time: {result['generation_time_seconds']:.2f} seconds")
            result_file = output_dir / "single_inference_result.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            if args.save_formatted:
                formatted_report = inference_engine.format_clinical_report(result['report'])
                formatted_file = output_dir / "formatted_report.txt"
                with open(formatted_file, 'w') as f:
                    f.write(formatted_report)
                logger.info(f"Formatted report saved: {formatted_file}")
        else:
            if args.image_dir:
                image_paths = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                    image_paths.extend(Path(args.image_dir).glob(ext))
                logger.info(f"Found {len(image_paths)} images in {args.image_dir}")
            else:
                image_paths = [args.image_path]
            if not image_paths:
                logger.error("No images found for processing")
                return 1
            batch_kwargs = {k: v for k, v in generation_kwargs.items() if k not in ['prompt', 'return_confidence']}
            logger.info(f"Starting batch processing of {len(image_paths)} images...")
            results = inference_engine.batch_inference(
                image_paths=image_paths,
                batch_size=args.batch_size,
                **batch_kwargs
            )
            logger.info(f"Processed {len(results)} images successfully")
            batch_results_file = output_dir / "batch_inference_results.json"
            with open(batch_results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Batch results saved: {batch_results_file}")
            avg_time = sum(r['generation_time_seconds'] for r in results) / len(results)
            total_time = sum(r['generation_time_seconds'] for r in results)
            logger.info("BATCH PROCESSING SUMMARY:")
            logger.info(f"  Total images processed: {len(results)}")
            logger.info(f"  Average time per image: {avg_time:.2f} seconds")
            logger.info(f"  Total processing time: {total_time:.2f} seconds")
        logger.info("=" * 80)
        logger.info("INFERENCE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
