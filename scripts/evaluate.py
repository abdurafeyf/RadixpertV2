#!/usr/bin/env python3

import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from ..models.radixpert import Radixpert, RadixpertConfig
from ..data.datasets import create_multi_stage_datasets
from ..evaluation.metrics import RadixpertEvaluator, create_evaluator
from ..utils.device_utils import setup_device, get_memory_info
from ..utils.logging_utils import setup_comprehensive_logging
from ..utils.checkpoint_utils import CheckpointManager
from ..utils.visualization import create_visualizer


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Radixpert model performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--model_config", type=str, default=None, help="Path to model configuration file")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory containing evaluation datasets")
    parser.add_argument("--dataset_type", type=str, choices=["padchest", "roco_v2", "both"], default="both", help="Which dataset to evaluate on")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test", help="Dataset split to evaluate")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate (None for all)")
    parser.add_argument("--batch_size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--generation_config", type=str, default=None, help="JSON file with text generation parameters")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum generation length")
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams for beam search")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Directory to save evaluation results")
    parser.add_argument("--save_predictions", action="store_true", help="Save generated predictions")
    parser.add_argument("--save_visualizations", action="store_true", help="Save attention visualizations")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"], help="Device to use for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    return parser.parse_args()


def load_model_and_config(model_path: str, model_config_path: Optional[str], device: torch.device, logger: logging.Logger) -> Radixpert:
    logger.info(f"Loading model from: {model_path}")
    checkpoint_manager = CheckpointManager(Path(model_path).parent)
    if model_config_path:
        with open(model_config_path, 'r') as f:
            config_dict = json.load(f)
        model_config = RadixpertConfig(**config_dict)
    else:
        model_config = RadixpertConfig()
    model = Radixpert(model_config).to(device)
    checkpoint_manager.load_checkpoint(model_path, model, device=device)
    model.eval()
    logger.info("Model loaded successfully")
    return model


def setup_evaluation_data(args: argparse.Namespace, logger: logging.Logger) -> Dict[str, Any]:
    logger.info("Setting up evaluation datasets...")
    if args.dataset_type == "both":
        padchest_root = Path(args.data_root) / "padchest"
        roco_root = Path(args.data_root) / "roco_v2"
        stage_datasets = create_multi_stage_datasets(
            padchest_root=str(padchest_root),
            roco_root=str(roco_root),
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        eval_datasets = stage_datasets[3]
    else:
        from data.datasets import create_padchest_config, create_roco_config, PadChestDataset, ROCOv2Dataset
        from torch.utils.data import DataLoader
        if args.dataset_type == "padchest":
            config = create_padchest_config(args.data_root)
            dataset = PadChestDataset(config, split=args.split)
        else:
            config = create_roco_config(args.data_root)
            dataset = ROCOv2Dataset(config, split=args.split)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        eval_datasets = {"single": dataloader}
    logger.info(f"Evaluation datasets ready: {list(eval_datasets.keys())}")
    return eval_datasets


def run_comprehensive_evaluation(model: Radixpert, eval_datasets: Dict[str, Any], args: argparse.Namespace, device: torch.device, logger: logging.Logger) -> Dict[str, Dict[str, float]]:
    logger.info("Starting comprehensive evaluation...")
    evaluator = create_evaluator(clinical_bert_model="emilyalsentzer/Bio_ClinicalBERT")
    visualizer = create_visualizer(f"{args.output_dir}/visualizations") if args.save_visualizations else None
    all_results = {}
    all_predictions = {}
    for dataset_name, dataloader in eval_datasets.items():
        logger.info(f"Evaluating on {dataset_name} dataset...")
        predictions = []
        references = []
        attention_maps = []
        with torch.no_grad():
            sample_count = 0
            for batch_idx, batch in enumerate(dataloader):
                images = batch['images'].to(device)
                generated_reports = model.generate_report(
                    images=images,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    num_beams=args.num_beams
                )
                predictions.extend(generated_reports)
                references.extend(batch['texts'])
                if visualizer and batch_idx < 5:
                    with torch.enable_grad():
                        outputs = model(images=images, return_fusion_weights=True)
                        if 'fusion_weights' in outputs:
                            attention_maps.append(outputs['fusion_weights'])
                sample_count += len(generated_reports)
                if args.max_samples and sample_count >= args.max_samples:
                    predictions = predictions[:args.max_samples]
                    references = references[:args.max_samples]
                    break
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Processed {sample_count} samples...")
        logger.info(f"Generated {len(predictions)} predictions for {dataset_name}")
        metrics = evaluator.compute_metrics(predictions, references)
        all_results[dataset_name] = metrics
        if args.save_predictions:
            predictions_data = {
                'predictions': predictions,
                'references': references,
                'metrics': metrics
            }
            predictions_file = Path(args.output_dir) / f"predictions_{dataset_name}.json"
            with open(predictions_file, 'w') as f:
                json.dump(predictions_data, f, indent=2, default=str)
            logger.info(f"Predictions saved: {predictions_file}")
        if visualizer and attention_maps:
            try:
                first_image = images[0] if 'images' in locals() else None
                first_attention = attention_maps[0] if attention_maps else None
                if first_image is not None and first_attention is not None:
                    visualizer.visualize_attention_maps(first_image, first_attention, predictions[0].split()[:20], f"attention_{dataset_name}.png")
                visualizer.analyze_generated_reports(
                    predictions[:100],
                    references[:100],
                    metrics,
                    f"report_analysis_{dataset_name}.png"
                )
            except Exception as e:
                logger.warning(f"Visualization failed: {e}")
        logger.info(f"\n{dataset_name.upper()} EVALUATION RESULTS:")
        evaluator.print_evaluation_report(metrics)
    return all_results


def save_evaluation_results(results: Dict[str, Dict[str, float]], args: argparse.Namespace, logger: logging.Logger):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        serializable_results = {}
        for dataset, metrics in results.items():
            serializable_results[dataset] = {k: float(v) if torch.is_tensor(v) else v for k, v in metrics.items()}
        json.dump(serializable_results, f, indent=2)
    logger.info(f"Detailed results saved: {results_file}")
    summary_file = output_dir / "evaluation_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("RADIXPERT EVALUATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Evaluation split: {args.split}\n")
        f.write(f"Max samples: {args.max_samples or 'All'}\n\n")
        for dataset, metrics in results.items():
            f.write(f"{dataset.upper()} DATASET:\n")
            f.write("-" * 30 + "\n")
            key_metrics = ['bleu_4', 'cider', 'radcliq_v1', 'radgraph_f1', 'clinical_accuracy']
            for metric in key_metrics:
                if metric in metrics:
                    f.write(f"  {metric.replace('_', ' ').title()}: {metrics[metric]:.4f}\n")
            f.write("\n")
    logger.info(f"Summary report saved: {summary_file}")


def main():
    args = parse_arguments()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_comprehensive_logging(output_dir=args.output_dir, log_level=args.log_level, experiment_name="radixpert_evaluation")
    logger.info("=" * 80)
    logger.info("RADIXPERT MODEL EVALUATION STARTED")
    logger.info("=" * 80)
    try:
        device = setup_device(logger) if args.device == "auto" else torch.device(args.device)
        memory_info = get_memory_info(device)
        logger.info(f"Device memory info: {memory_info}")
        model = load_model_and_config(args.model_path, args.model_config, device, logger)
        eval_datasets = setup_evaluation_data(args, logger)
        results = run_comprehensive_evaluation(model, eval_datasets, args, device, logger)
        save_evaluation_results(results, args, logger)
        logger.info("=" * 80)
        logger.info("EVALUATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        for dataset, metrics in results.items():
            logger.info(f"\n{dataset.upper()} FINAL METRICS:")
            logger.info(f"  BLEU-4: {metrics.get('bleu_4', 0):.4f}")
            logger.info(f"  CIDEr: {metrics.get('cider', 0):.4f}")
            logger.info(f"  RadCliQ-v1: {metrics.get('radcliq_v1', 0):.4f}")
        logger.info(f"\nAll results saved to: {args.output_dir}")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
