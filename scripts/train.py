#!/usr/bin/env python3

import os
import sys
import argparse
import logging
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
import torch.multiprocessing as mp
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from models.radixpert import Radixpert, RadixpertConfig
from training.multi_stage_trainer import MultiStageTrainer, create_multi_stage_trainer
from training.losses import MultiStageLoss, LossConfig
from training.optimization import OptimizationConfig
from data.datasets import create_multi_stage_datasets, create_padchest_config, create_roco_config
from evaluation.metrics import RadixpertEvaluator, create_evaluator
from config.training_config import TrainingConfig, create_stage_configs
from utils.logging import setup_comprehensive_logging, log_system_info, log_model_info
from utils.checkpoint_utils import CheckpointManager, find_latest_checkpoint
from utils.device_utils import setup_device, get_memory_info


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Radixpert model for radiology generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--padchest_root", type=str, required=True, help="Root of PadChest dataset")
    parser.add_argument("--roco_root", type=str, required=True, help="Root of ROCO dataset")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct", help="Base LLaMA model")
    parser.add_argument("--vision_backbone", type=str, default="efficientnet_b7", choices=["efficientnet_b7", "resnet50", "vit_base", "swin_transformer"], help="Vision backbone")
    parser.add_argument("--output_dir", type=str, default="./radixpert_training", help="Checkpoint/log directory")
    parser.add_argument("--stages", type=str, default="1,2,3", help="Training stages to run, e.g. 1,2,3")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--num_epochs", type=int, nargs=3, default=[10, 15, 20], help="Epochs per stage")
    parser.add_argument("--learning_rate", type=float, nargs=3, default=[5e-5, 3e-5, 1e-5], help="Learning rates per stage")
    parser.add_argument("--lora_rank", type=int, nargs=3, default=[8, 12, 16], help="LoRA ranks per stage")
    parser.add_argument("--resume_from", type=str, default=None, help="Checkpoint path for resuming")
    parser.add_argument("--auto_resume", action="store_true", help="Auto-resume from latest checkpoint")
    parser.add_argument("--eval_every_n", type=int, default=2, help="Evaluate every n epochs")
    parser.add_argument("--save_every_n", type=int, default=5, help="Save every n epochs")
    parser.add_argument("--max_eval_samples", type=int, default=1000, help="Max eval samples")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data workers")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--wandb_project", type=str, default="radixpert", help="Weights & Biases project")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    parser.add_argument("--config", type=str, default=None, help="YAML config file path")
    parser.add_argument("--use_sam", action="store_true", help="Use Sharpness-Aware Minimizer")
    parser.add_argument("--use_lookahead", action="store_true", help="Use Lookahead optimizer")
    parser.add_argument("--compile_model", action="store_true", help="Use torch.compile for training")
    return parser.parse_args()


def load_config_file(config_path: str) -> Dict[str, Any]:
    with open(config_path) as f:
        if config_path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        return json.load(f)


def create_configs(args):
    stages = [int(s) for s in args.stages.split(",")]
    model_cfg = RadixpertConfig(
        llama_model_name=args.model_name,
        max_length=2048,
        fusion_layers=3,
        enable_clinical_validation=True,
        lora_rank=args.lora_rank[0]
    )
    stage_cfgs = {}
    for idx, stage in enumerate([1, 2, 3]):
        stage_cfgs[stage] = {
            "stage": stage,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs[idx] if idx < len(args.num_epochs) else 10,
            "learning_rate": args.learning_rate[idx] if idx < len(args.learning_rate) else 1e-5,
            "lora_rank": args.lora_rank[idx] if idx < len(args.lora_rank) else 16,
            "eval_every_n": args.eval_every_n,
            "save_every_n": args.save_every_n,
        }
    optim_cfg = OptimizationConfig(
        use_mixed_precision=args.mixed_precision,
        gradient_clipping=1.0,
        use_sam=args.use_sam,
        use_lookahead=args.use_lookahead,
    )
    loss_cfg = LossConfig(
        label_smoothing=0.1,
        temperature=0.07,
        clinical_weight=0.5,
    )
    return {
        "model_config": model_cfg,
        "stage_configs": stage_cfgs,
        "optimization_config": optim_cfg,
        "loss_config": loss_cfg,
        "stages_to_run": stages,
        "training_args": args,
    }


def setup_data_loaders(args, logger):
    logger.info("Setting up data loaders...")
    padchest_cfg = create_padchest_config(
        args.padchest_root, image_size=(512, 512), max_text_length=512, use_medical_preprocessing=True
    )
    roco_cfg = create_roco_config(
        args.roco_root, image_size=(512, 512), max_text_length=512, use_medical_preprocessing=True
    )
    try:
        datasets = create_multi_stage_datasets(
            padchest_root=args.padchest_root,
            roco_root=args.roco_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        logger.info("Data loaders created successfully")
        for stage_num, ds in datasets.items():
            stats = ds.get_dataset_statistics()
            logger.info(f"Stage {stage_num} datasets: {stats}")
        return datasets
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        raise


def setup_trainer(configs, datasets, device, logger):
    logger.info("Setting up trainer and model...")
    model_cfg = configs["model_config"]
    stage_cfgs = configs["stage_configs"]
    args = configs["training_args"]
    try:
        train_loaders = {}
        val_loaders = {}
        for stage_num in configs["stages_to_run"]:
            if stage_num in datasets:
                train_loaders[stage_num] = datasets[stage_num].get_combined_dataloader("train")
                val_loaders[stage_num] = datasets[stage_num].get_combined_dataloader("val")
        trainer = MultiStageTrainer(
            model_config=model_cfg,
            stage_configs=stage_cfgs,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            device=device,
            output_dir=args.output_dir,
            log_level=args.log_level,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
        )
        logger.info("Trainer created successfully")
        if trainer.model is not None:
            log_model_info(trainer.model, logger)
        return trainer
    except Exception as e:
        logger.error(f"Failed to set up trainer: {e}")
        raise


def run_training(trainer, configs, logger):
    args = configs["training_args"]
    stages = configs["stages_to_run"]
    logger.info("Starting multi-stage training...")
    logger.info(f"Stages to run: {stages}")
    start_stage = min(stages)
    resume_checkpoint = None
    if args.resume_from:
        resume_checkpoint = args.resume_from
        logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
    elif args.auto_resume:
        checkpoint_mgr = CheckpointManager(args.output_dir)
        latest = find_latest_checkpoint(args.output_dir)
        if latest:
            resume_checkpoint = latest
            logger.info(f"Auto-resuming from: {resume_checkpoint}")
    try:
        results = trainer.train_all_stages(
            start_stage=start_stage, end_stage=max(stages), load_checkpoint=resume_checkpoint
        )
        logger.info("Training completed successfully")
        for stg, metrics in results.items():
            logger.info(f"Stage {stg} final metrics: {metrics}")
        return results
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        emergency_path = Path(trainer.output_dir) / "emergency_checkpoint.pt"
        trainer.save_checkpoint(emergency_path, trainer.current_stage, {})
        logger.info(f"Emergency checkpoint saved at {emergency_path}")
        raise
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def run_final_evaluation(trainer, datasets, args, logger):
    logger.info("Running final evaluation...")
    evaluator = create_evaluator(clinical_bert_model="emilyalsentzer/Bio_ClinicalBERT")
    final_metrics = {}
    for stage_num, dataset in datasets.items():
        logger.info(f"Evaluating stage {stage_num} test set...")
        try:
            test_loader = dataset.get_combined_dataloader("test")
            metrics = evaluator.evaluate(
                model=trainer.model, dataloader=test_loader, device=trainer.device, max_samples=args.max_eval_samples
            )
            final_metrics[f"stage_{stage_num}"] = metrics
            logger.info(f"Stage {stage_num} evaluation metrics:")
            evaluator.print_evaluation_report(metrics)
        except Exception as e:
            logger.error(f"Failed evaluation at stage {stage_num}: {e}")
            continue
    output_dir = Path(args.output_dir)
    metrics_file = output_dir / "final_evaluation_metrics.json"
    with open(metrics_file, "w") as f:
        serializable = {}
        for k, v in final_metrics.items():
            if isinstance(v, dict):
                serializable[k] = {mk: float(mv) if torch.is_tensor(mv) else mv for mk, mv in v.items()}
            else:
                serializable[k] = float(v) if torch.is_tensor(v) else v
        json.dump(serializable, f, indent=2)
    logger.info(f"Final evaluation metrics saved: {metrics_file}")
    return final_metrics


def main():
    args = parse_arguments()
    if args.config:
        cfg = load_config_file(args.config)
        for key, val in cfg.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, val)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_comprehensive_logging(
        output_dir=str(output_dir), log_level=args.log_level, experiment_name=f"radixpert_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    logger.info("=" * 80)
    logger.info("Radixpert training started")
    logger.info("=" * 80)
    try:
        device = setup_device(logger) if args.device == "auto" else torch.device(args.device)
        logger.info(f"Using device: {device}")
        log_system_info(logger)
        configs = create_configs(args)
        datasets = setup_data_loaders(args, logger)
        trainer = setup_trainer(configs, datasets, device, logger)
        if args.compile_model and hasattr(torch, "compile"):
            logger.info("Compiling model for faster training")
            trainer.model = torch.compile(trainer.model)
        train_res = run_training(trainer, configs, logger)
        eval_res = run_final_evaluation(trainer, datasets, args, logger)
        logger.info("Training and evaluation completed successfully")
        if "stage_3" in eval_res:
            m = eval_res["stage_3"]
            logger.info("Stage 3 key metrics:")
            logger.info(f"BLEU-4: {m.get('bleu_4', 0):.4f}, CIDEr: {m.get('cider',0):.4f}, RadCliQ-v1: {m.get('radcliq_v1',0):.4f}")
        logger.info(f"Results saved in {args.output_dir}")
        return 0
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    if mp.is_initialized():
        pass
    else:
        mp.set_start_method("spawn", force=True)
    sys.exit(main())
