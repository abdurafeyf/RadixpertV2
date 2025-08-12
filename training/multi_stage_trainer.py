import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import time
import wandb
from pathlib import Path
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from enum import Enum

from ..models.radixpert import Radixpert, RadixpertConfig
from ..evaluation.metrics import RadixpertEvaluator
from ..utils.logging_utils import setup_comprehensive_logging
from ..utils.checkpoint_utils import CheckpointManager
from .losses import MultiStageLoss
from .optimization import create_optimizer, create_scheduler


class TrainingStage(Enum):
    PRE_ALIGNMENT = 1
    DOMAIN_ADAPTATION = 2
    TASK_SPECIFIC = 3


@dataclass
class StageConfig:
    stage: int
    name: str
    description: str
    datasets: List[str]
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: int
    loss_weights: Dict[str, float]
    gradient_clip_norm: float = 1.0
    use_amp: bool = True
    eval_every_n_steps: int = 500
    save_every_n_steps: int = 1000
    freeze_previous_stages: bool = True
    progressive_unfreezing: bool = True


class MultiStageTrainer:
    def __init__(
        self,
        model_config: RadixpertConfig,
        stage_configs: Dict[int, StageConfig],
        train_dataloaders: Dict[int, DataLoader],
        val_dataloaders: Dict[int, DataLoader],
        device: torch.device = None,
        output_dir: str = "./radixpert_checkpoints",
        log_level: str = "INFO",
        use_wandb: bool = True,
        wandb_project: str = "radixpert",
    ):
        self.model_config = model_config
        self.stage_configs = stage_configs
        self.train_dataloaders = train_dataloaders
        self.val_dataloaders = val_dataloaders
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.current_stage = 1
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_comprehensive_logging("MultiStageTrainer", log_level)
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project=wandb_project,
                config={
                    "model_config": asdict(model_config),
                    "stage_configs": {k: asdict(v) for k, v in stage_configs.items()}
                }
            )
        self.checkpoint_manager = CheckpointManager(self.output_dir)
        self.evaluator = RadixpertEvaluator()
        self.loss_calculator = MultiStageLoss()
        self.global_step = 0
        self.best_metrics = {}
        self.training_history = defaultdict(list)
        self.logger.info(f"MultiStageTrainer initialized with {len(stage_configs)} stages")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Device: {self.device}")

    def train_all_stages(self, start_stage: int = 1, end_stage: int = 3, load_checkpoint: Optional[str] = None):
        self.logger.info("="*80)
        self.logger.info("STARTING RADIXPERT MULTI-STAGE TRAINING")
        self.logger.info("="*80)
        if load_checkpoint:
            self.load_checkpoint(load_checkpoint)
            self.logger.info(f"Resumed from checkpoint: {load_checkpoint}")
        for stage_num in range(start_stage, end_stage + 1):
            if stage_num not in self.stage_configs:
                self.logger.warning(f"Stage {stage_num} configuration not found, skipping")
                continue
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"STARTING STAGE {stage_num}: {self.stage_configs[stage_num].name}")
            self.logger.info(f"{'='*60}")
            stage_metrics = self.train_stage(stage_num)
            self.logger.info(f"Stage {stage_num} completed with metrics: {stage_metrics}")
            if stage_num not in self.best_metrics or stage_metrics['val_loss'] < self.best_metrics[stage_num]['val_loss']:
                self.best_metrics[stage_num] = stage_metrics
            stage_checkpoint_path = self.output_dir / f"stage_{stage_num}_final.pt"
            self.save_checkpoint(stage_checkpoint_path, stage_num, stage_metrics)
            self.logger.info(f"Stage {stage_num} checkpoint saved: {stage_checkpoint_path}")
        self.final_evaluation()
        self.logger.info("="*80)
        self.logger.info("RADIXPERT MULTI-STAGE TRAINING COMPLETED")
        self.logger.info("="*80)
        return self.best_metrics

    def train_stage(self, stage_num: int) -> Dict[str, float]:
        stage_config = self.stage_configs[stage_num]
        self.current_stage = stage_num
        if self.model is None:
            self.model = self._initialize_model_for_stage(stage_num)
        else:
            self._update_model_for_stage(stage_num)
        optimizer = self._create_optimizer(stage_config)
        scheduler = self._create_scheduler(optimizer, stage_config)
        scaler = torch.cuda.amp.GradScaler() if stage_config.use_amp else None
        train_loader = self.train_dataloaders[stage_num]
        val_loader = self.val_dataloaders[stage_num]
        stage_metrics = {
            'train_loss': 0.0,
            'val_loss': 0.0,
            'learning_rate': 0.0,
            'grad_norm': 0.0,
            'epoch_time': 0.0
        }
        self.logger.info(f"Training configuration for Stage {stage_num}:")
        self.logger.info(f"  - Datasets: {stage_config.datasets}")
        self.logger.info(f"  - Batch size: {stage_config.batch_size}")
        self.logger.info(f"  - Learning rate: {stage_config.learning_rate}")
        self.logger.info(f"  - Epochs: {stage_config.num_epochs}")
        self.logger.info(f"  - LoRA rank: {stage_config.lora_rank}")
        best_val_loss = float('inf')
        for epoch in range(stage_config.num_epochs):
            epoch_start_time = time.time()
            train_metrics = self._train_epoch(train_loader, optimizer, scheduler, scaler, stage_config)
            val_metrics = self._validate_epoch(val_loader, stage_config)
            epoch_time = time.time() - epoch_start_time
            current_lr = optimizer.param_groups[0]['lr']
            self.logger.info(
                f"Stage {stage_num}, Epoch {epoch+1}/{stage_config.num_epochs}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"LR: {current_lr:.2e}, "
                f"Time: {epoch_time:.1f}s"
            )
            stage_metrics.update({
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'learning_rate': current_lr,
                'grad_norm': train_metrics.get('grad_norm', 0.0),
                'epoch_time': epoch_time
            })
            if self.use_wandb:
                wandb.log({
                    f"stage_{stage_num}/train_loss": train_metrics['loss'],
                    f"stage_{stage_num}/val_loss": val_metrics['loss'],
                    f"stage_{stage_num}/learning_rate": current_lr,
                    f"stage_{stage_num}/epoch": epoch + 1,
                    "global_step": self.global_step
                })
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_checkpoint_path = self.output_dir / f"stage_{stage_num}_best.pt"
                self.save_checkpoint(best_checkpoint_path, stage_num, stage_metrics)
            if (epoch + 1) % 5 == 0:
                checkpoint_path = self.output_dir / f"stage_{stage_num}_epoch_{epoch+1}.pt"
                self.save_checkpoint(checkpoint_path, stage_num, stage_metrics)
        return stage_metrics

    def _train_epoch(self, train_loader, optimizer, scheduler, scaler, stage_config):
        self.model.train()
        total_loss = 0.0
        total_grad_norm = 0.0
        num_batches = 0
        for batch_idx, batch in enumerate(train_loader):
            batch = self._move_batch_to_device(batch)
            optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        images=batch['images'],
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                    )
                    loss_dict = self.loss_calculator.compute_loss(outputs, batch, self.current_stage, stage_config.loss_weights)
                    loss = loss_dict['total_loss']
                scaler.scale(loss).backward()
                if stage_config.gradient_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), stage_config.gradient_clip_norm)
                    total_grad_norm += grad_norm.item()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = self.model(
                    images=batch['images'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                loss_dict = self.loss_calculator.compute_loss(outputs, batch, self.current_stage, stage_config.loss_weights)
                loss = loss_dict['total_loss']
                loss.backward()
                if stage_config.gradient_clip_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), stage_config.gradient_clip_norm)
                    total_grad_norm += grad_norm.item()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            if batch_idx % 100 == 0:
                self.logger.debug(f"Batch {batch_idx}/{len(train_loader)}: Loss: {loss.item():.4f}")
            if self.global_step % stage_config.eval_every_n_steps == 0:
                val_metrics = self._validate_epoch(self.val_dataloaders[self.current_stage], stage_config)
                self.logger.info(f"Step {self.global_step}: Val Loss: {val_metrics['loss']:.4f}")
                self.model.train()
        return {'loss': total_loss / num_batches, 'grad_norm': total_grad_norm / num_batches if num_batches > 0 else 0.0}

    def _validate_epoch(self, val_loader, stage_config):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = self._move_batch_to_device(batch)
                outputs = self.model(
                    images=batch['images'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                loss_dict = self.loss_calculator.compute_loss(outputs, batch, self.current_stage, stage_config.loss_weights)
                total_loss += loss_dict['total_loss'].item()
                num_batches += 1
        return {'loss': total_loss / num_batches if num_batches > 0 else 0.0}

    def _initialize_model_for_stage(self, stage_num: int) -> Radixpert:
        stage_config = self.stage_configs[stage_num]
        model_config = self.model_config
        model_config.stage = stage_num
        model_config.lora_rank = stage_config.lora_rank
        model_config.lora_alpha = stage_config.lora_alpha
        model_config.lora_dropout = stage_config.lora_dropout
        model = Radixpert(model_config)
        model = model.to(self.device)
        self.logger.info(f"Initialized model for Stage {stage_num}")
        trainable_params = model.get_trainable_parameters()
        self.logger.info(f"Trainable parameters: {trainable_params}")
        return model

    def _update_model_for_stage(self, stage_num: int):
        if hasattr(self.model, 'msa_lora_manager'):
            self.model.msa_lora_manager.update_stage(stage_num)
        self.model.config.stage = stage_num
        self.model._configure_training_stage()
        self.logger.info(f"Updated model for Stage {stage_num}")

    def _create_optimizer(self, stage_config: StageConfig) -> optim.Optimizer:
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if stage_config.stage == 1:
            optimizer = optim.AdamW(trainable_params, lr=stage_config.learning_rate, weight_decay=stage_config.weight_decay, betas=(0.9, 0.95))
        elif stage_config.stage == 2:
            optimizer = optim.AdamW(trainable_params, lr=stage_config.learning_rate, weight_decay=stage_config.weight_decay, betas=(0.9, 0.98))
        else:
            optimizer = optim.AdamW(trainable_params, lr=stage_config.learning_rate, weight_decay=stage_config.weight_decay, betas=(0.9, 0.999))
        return optimizer

    def _create_scheduler(self, optimizer: optim.Optimizer, stage_config: StageConfig) -> any:
        total_steps = len(self.train_dataloaders[stage_config.stage]) * stage_config.num_epochs
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=total_steps // 4, T_mult=2, eta_min=stage_config.learning_rate * 0.01)
        return scheduler

    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def save_checkpoint(self, path: Union[str, Path], stage_num: int, metrics: Dict[str, float]):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model_config,
            'stage_num': stage_num,
            'global_step': self.global_step,
            'metrics': metrics,
            'training_history': dict(self.training_history),
            'best_metrics': self.best_metrics
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: Union[str, Path]):
        checkpoint = torch.load(path, map_location=self.device)
        if self.model is None:
            self.model = self._initialize_model_for_stage(checkpoint['stage_num'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.global_step = checkpoint['global_step']
        self.current_stage = checkpoint['stage_num']
        self.training_history = defaultdict(list, checkpoint.get('training_history', {}))
        self.best_metrics = checkpoint.get('best_metrics', {})
        self.logger.info(f"Checkpoint loaded from: {path}")
        self.logger.info(f"Resumed at Stage {self.current_stage}, Step {self.global_step}")

    def final_evaluation(self):
        self.logger.info("Starting final evaluation...")
        final_metrics = {}
        for stage_num, val_loader in self.val_dataloaders.items():
            if hasattr(self.model, 'msa_lora_manager'):
                self.model.msa_lora_manager.update_stage(stage_num)
            stage_metrics = self.evaluator.evaluate(self.model, val_loader, device=self.device)
            final_metrics[f'stage_{stage_num}'] = stage_metrics
            self.logger.info(f"Stage {stage_num} final metrics: {stage_metrics}")
        if self.use_wandb:
            wandb.log({"final_metrics": final_metrics})
        with open(self.output_dir / "final_metrics.json", "w") as f:
            json.dump(final_metrics, f, indent=2)
        return final_metrics


def create_multi_stage_trainer(
    model_config: RadixpertConfig,
    train_datasets: Dict[int, Any],
    val_datasets: Dict[int, Any],
    output_dir: str = "./radixpert_training",
    **kwargs
) -> MultiStageTrainer:
    default_stage_configs = {
        1: StageConfig(stage=1, name="Cross-Modal Pre-alignment", description="Initial alignment using ROCO v2 dataset", datasets=["roco_v2"], batch_size=16, num_epochs=10, learning_rate=5e-5, weight_decay=1e-4, warmup_steps=1000, lora_rank=8, lora_alpha=16, lora_dropout=0.1, loss_weights={"generation":1.0,"contrastive":0.5,"regularization":0.1}),
        2: StageConfig(stage=2, name="Domain Adaptation", description="Domain adaptation using ROCO v2 + PadChest", datasets=["roco_v2","padchest"], batch_size=12, num_epochs=15, learning_rate=3e-5, weight_decay=1e-4, warmup_steps=1500, lora_rank=12, lora_alpha=24, lora_dropout=0.1, loss_weights={"generation":1.0,"contrastive":0.7,"clinical":0.3,"regularization":0.1}),
        3: StageConfig(stage=3,name="Task-Specific Fine-tuning",description="Final fine-tuning on complete combined dataset",datasets=["roco_v2","padchest","mimic_cxr"],batch_size=8,num_epochs=20,learning_rate=1e-5,weight_decay=1e-4,warmup_steps=2000,lora_rank=16,lora_alpha=32,lora_dropout=0.1,loss_weights={"generation":1.0,"contrastive":0.8,"clinical":0.5,"fusion":0.3,"regularization":0.1}),
    }
    train_dataloaders = {}
    val_dataloaders = {}
    for stage_num, dataset in train_datasets.items():
        stage_config = default_stage_configs[stage_num]
        train_dataloaders[stage_num] = DataLoader(dataset, batch_size=stage_config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    for stage_num, dataset in val_datasets.items():
        stage_config = default_stage_configs[stage_num]
        val_dataloaders[stage_num] = DataLoader(dataset, batch_size=stage_config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return MultiStageTrainer(model_config=model_config, stage_configs=default_stage_configs,
                             train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders, output_dir=output_dir, **kwargs)


if __name__ == "__main__":
    from ..models.radixpert import RadixpertConfig
    model_config = RadixpertConfig(llama_model_name="meta-llama/Llama-3.2-11B-Vision-Instruct", max_length=2048, lora_rank=16, lora_alpha=32, fusion_layers=3, fusion_dim=4096)
    train_datasets = {1: None, 2: None, 3: None}  # replace with actual datasets
    val_datasets = {1: None, 2: None, 3: None}    # replace with actual datasets
    trainer = create_multi_stage_trainer(model_config, train_datasets, val_datasets, output_dir="./radixpert_checkpoints", use_wandb=True, wandb_project="radixpert-training")
    final_metrics = trainer.train_all_stages()
    print("Training completed!")
    print(f"Final metrics: {final_metrics}")
