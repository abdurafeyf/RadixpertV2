import torch
import torch.nn as nn
from pathlib import Path
import json
import logging
import shutil
from typing import Dict, List, Optional, Union, Any
from datetime import datetime


class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 5,
        save_best_only: bool = False,
        monitor_metric: str = "val_loss",
        mode: str = "min"
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.best_checkpoint_path = None
        self.checkpoint_history = self._load_checkpoint_history()
        self.logger = logging.getLogger("CheckpointManager")
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        stage: int,
        metrics: Dict[str, float],
        extra_state: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None
    ) -> str:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"checkpoint_stage{stage}_epoch{epoch}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = {
            'epoch': epoch,
            'stage': stage,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__
        }
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if extra_state:
            checkpoint['extra_state'] = extra_state
        if hasattr(model, 'config'):
            checkpoint['model_config'] = model.config
        try:
            temp_path = checkpoint_path.with_suffix('.tmp')
            torch.save(checkpoint, temp_path)
            shutil.move(str(temp_path), str(checkpoint_path))
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            self._update_checkpoint_history(checkpoint_path, metrics)
            current_metric = metrics.get(self.monitor_metric)
            if current_metric is not None:
                is_best = self._is_best_metric(current_metric)
                if is_best:
                    self.best_metric = current_metric
                    self.best_checkpoint_path = str(checkpoint_path)
                    best_path = self.checkpoint_dir / f"best_stage{stage}.pt"
                    shutil.copy2(str(checkpoint_path), str(best_path))
                    self.logger.info(f"New best checkpoint: {best_path} ({self.monitor_metric}: {current_metric:.4f})")
            if not self.save_best_only:
                self._cleanup_old_checkpoints()
            return str(checkpoint_path)
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        strict: bool = True
    ) -> Dict[str, Any]:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        try:
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            self.logger.info(f"Model state loaded from {checkpoint_path}")
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.logger.info("Optimizer state loaded")
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.logger.info("Scheduler state loaded")
            metadata = {
                'epoch': checkpoint.get('epoch', 0),
                'stage': checkpoint.get('stage', 1),
                'metrics': checkpoint.get('metrics', {}),
                'timestamp': checkpoint.get('timestamp', ''),
                'extra_state': checkpoint.get('extra_state', {})
            }
            self.logger.info(f"Checkpoint loaded successfully - Stage: {metadata['stage']}, Epoch: {metadata['epoch']}")
            return metadata
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            raise
    
    def find_latest_checkpoint(self, stage: Optional[int] = None) -> Optional[str]:
        pattern = "checkpoint_*.pt"
        if stage is not None:
            pattern = f"checkpoint_stage{stage}_*.pt"
        checkpoint_files = list(self.checkpoint_dir.glob(pattern))
        if not checkpoint_files:
            return None
        latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        return str(latest_checkpoint)
    
    def find_best_checkpoint(self, stage: Optional[int] = None) -> Optional[str]:
        if stage is not None:
            best_path = self.checkpoint_dir / f"best_stage{stage}.pt"
            if best_path.exists():
                return str(best_path)
        return self.best_checkpoint_path
    
    def _is_best_metric(self, current_metric: float) -> bool:
        if self.mode == 'min':
            return current_metric < self.best_metric
        else:
            return current_metric > self.best_metric
    
    def _update_checkpoint_history(self, checkpoint_path: Path, metrics: Dict[str, float]):
        history_entry = {
            'path': str(checkpoint_path),
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        self.checkpoint_history.append(history_entry)
        history_file = self.checkpoint_dir / "checkpoint_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.checkpoint_history, f, indent=2)
    
    def _load_checkpoint_history(self) -> List[Dict[str, Any]]:
        history_file = self.checkpoint_dir / "checkpoint_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    return json.load(f)
            except:
                self.logger.warning("Failed to load checkpoint history")
        return []
    
    def _cleanup_old_checkpoints(self):
        checkpoint_files = [
            f for f in self.checkpoint_dir.glob("checkpoint_*.pt")
            if not f.name.startswith("best_")
        ]
        if len(checkpoint_files) > self.max_checkpoints:
            checkpoint_files.sort(key=lambda p: p.stat().st_mtime)
            files_to_remove = checkpoint_files[:-self.max_checkpoints]
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                    self.logger.debug(f"Removed old checkpoint: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove checkpoint {file_path}: {e}")


def find_latest_checkpoint(checkpoint_dir: Union[str, Path]) -> Optional[str]:
    manager = CheckpointManager(checkpoint_dir)
    return manager.find_latest_checkpoint()


def safe_load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    device: Optional[torch.device] = None
) -> bool:
    try:
        manager = CheckpointManager(Path(checkpoint_path).parent)
        manager.load_checkpoint(checkpoint_path, model, device=device)
        return True
    except Exception as e:
        logging.getLogger("CheckpointUtils").error(f"Failed to load checkpoint: {e}")
        return False
