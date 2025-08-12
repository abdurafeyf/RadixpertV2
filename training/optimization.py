import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingWarmRestarts
import math
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
from enum import Enum
import warnings
import numpy as np


class OptimizerType(Enum):
    ADAMW = "adamw"
    ADAM = "adam"
    SGDM = "sgdm"
    ADAFACTOR = "adafactor"
    LION = "lion"


class SchedulerType(Enum):
    COSINE = "cosine"
    COSINE_RESTART = "cosine_restart"
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    EXPONENTIAL = "exponential"
    PLATEAU = "plateau"
    CUSTOM_MEDICAL = "custom_medical"


@dataclass
class OptimizationConfig:
    stage_configs: Dict[int, Dict[str, Any]] = None
    optimizer_type: OptimizerType = OptimizerType.ADAMW
    learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-8
    scheduler_type: SchedulerType = SchedulerType.COSINE_RESTART
    warmup_steps: int = 1000
    warmup_ratio: float = 0.1
    num_cycles: int = 1
    min_lr_ratio: float = 0.01
    use_gradient_checkpointing: bool = False
    use_mixed_precision: bool = True
    gradient_clipping: float = 1.0
    gradient_accumulation_steps: int = 1
    clinical_learning_rate_factor: float = 0.5
    vision_learning_rate_factor: float = 0.1
    text_learning_rate_factor: float = 1.0
    use_adafactor: bool = False
    adafactor_scale_parameter: bool = True
    adafactor_relative_step: bool = False
    use_sam: bool = False
    sam_rho: float = 0.05
    use_lookahead: bool = False
    lookahead_k: int = 5
    lookahead_alpha: float = 0.5

    def __post_init__(self):
        if self.stage_configs is None:
            self.stage_configs = {
                1: {
                    'learning_rate': 5e-5,
                    'weight_decay': 1e-4,
                    'betas': (0.9, 0.95),
                    'warmup_steps': 1000,
                    'scheduler_type': SchedulerType.LINEAR,
                    'gradient_clipping': 0.5,
                    'min_lr_ratio': 0.1
                },
                2: {
                    'learning_rate': 3e-5,
                    'weight_decay': 1e-4,
                    'betas': (0.9, 0.98),
                    'warmup_steps': 1500,
                    'scheduler_type': SchedulerType.COSINE_RESTART,
                    'gradient_clipping': 1.0,
                    'min_lr_ratio': 0.05
                },
                3: {
                    'learning_rate': 1e-5,
                    'weight_decay': 1e-4,
                    'betas': (0.9, 0.999),
                    'warmup_steps': 2000,
                    'scheduler_type': SchedulerType.COSINE,
                    'gradient_clipping': 1.0,
                    'min_lr_ratio': 0.01
                }
            }


class LayerWiseLearningRateOptimizer:
    def __init__(self, model: torch.nn.Module, config: OptimizationConfig, stage: int = 1):
        self.model = model
        self.config = config
        self.stage = stage
        self.stage_config = config.stage_configs.get(stage, config.stage_configs[3])
        self.base_lr = self.stage_config.get('learning_rate', config.learning_rate)

    def get_parameter_groups(self) -> List[Dict[str, Any]]:
        vision_params, text_params, fusion_params, lora_params, other_params = [], [], [], [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if self._is_vision_parameter(name):
                vision_params.append(param)
            elif self._is_text_parameter(name):
                text_params.append(param)
            elif self._is_fusion_parameter(name):
                fusion_params.append(param)
            elif self._is_lora_parameter(name):
                lora_params.append(param)
            else:
                other_params.append(param)
        param_groups = []
        if vision_params:
            param_groups.append({
                'params': vision_params,
                'lr': self.base_lr * self.config.vision_learning_rate_factor,
                'name': 'vision_encoder',
                'weight_decay': self.stage_config.get('weight_decay', self.config.weight_decay) * 0.1
            })
        if text_params:
            param_groups.append({
                'params': text_params,
                'lr': self.base_lr * self.config.text_learning_rate_factor,
                'name': 'text_decoder',
                'weight_decay': self.stage_config.get('weight_decay', self.config.weight_decay)
            })
        if fusion_params:
            param_groups.append({
                'params': fusion_params,
                'lr': self.base_lr,
                'name': 'fusion_layers',
                'weight_decay': self.stage_config.get('weight_decay', self.config.weight_decay)
            })
        if lora_params:
            param_groups.append({
                'params': lora_params,
                'lr': self.base_lr * 1.5,
                'name': 'lora_adapters',
                'weight_decay': self.stage_config.get('weight_decay', self.config.weight_decay) * 0.5
            })
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.base_lr,
                'name': 'other_params',
                'weight_decay': self.stage_config.get('weight_decay', self.config.weight_decay)
            })
        return param_groups

    def _is_vision_parameter(self, name: str) -> bool:
        indicators = ['vision_encoder', 'backbone', 'conv', 'visual', 'image_encoder',
                      'patch_embed', 'blocks', 'norm1', 'norm2', 'attn']
        return any(ind in name.lower() for ind in indicators)

    def _is_text_parameter(self, name: str) -> bool:
        indicators = ['llama', 'transformer', 'embed_tokens', 'layers', 'self_attn',
                      'mlp', 'input_layernorm', 'post_attention_layernorm', 'lm_head']
        return any(ind in name.lower() for ind in indicators)

    def _is_fusion_parameter(self, name: str) -> bool:
        indicators = ['hcf_fusion', 'fusion', 'cross_modal', 'attention', 'combiner',
                      'gating', 'level', 'hierarchical']
        return any(ind in name.lower() for ind in indicators)

    def _is_lora_parameter(self, name: str) -> bool:
        indicators = ['lora_a', 'lora_b', 'lora_', 'adapter']
        return any(ind in name.lower() for ind in indicators)


class MedicalOptimizer:
    def __init__(self, model: torch.nn.Module, config: OptimizationConfig, stage: int = 1, total_steps: Optional[int] = None):
        self.model = model
        self.config = config
        self.stage = stage
        self.total_steps = total_steps
        self.stage_config = config.stage_configs.get(stage, config.stage_configs[3])
        self.layer_wise_optimizer = LayerWiseLearningRateOptimizer(model, config, stage)
        self.param_groups = self.layer_wise_optimizer.get_parameter_groups()
        self.optimizer = self._create_base_optimizer()
        self.scheduler = self._create_scheduler()
        self.sam_optimizer = SAM(self.optimizer, rho=config.sam_rho) if config.use_sam else None
        self.lookahead_optimizer = Lookahead(self.optimizer, k=config.lookahead_k, alpha=config.lookahead_alpha) if config.use_lookahead else None

    def _create_base_optimizer(self) -> torch.optim.Optimizer:
        optimizer_type = self.stage_config.get('optimizer_type', self.config.optimizer_type)
        if optimizer_type == OptimizerType.ADAMW:
            return optim.AdamW(
                self.param_groups,
                lr=self.stage_config.get('learning_rate', self.config.learning_rate),
                betas=self.stage_config.get('betas', self.config.betas),
                eps=self.config.eps,
                weight_decay=self.stage_config.get('weight_decay', self.config.weight_decay)
            )
        elif optimizer_type == OptimizerType.ADAM:
            return optim.Adam(
                self.param_groups,
                lr=self.stage_config.get('learning_rate', self.config.learning_rate),
                betas=self.stage_config.get('betas', self.config.betas),
                eps=self.config.eps,
                weight_decay=self.stage_config.get('weight_decay', self.config.weight_decay)
            )
        elif optimizer_type == OptimizerType.SGDM:
            return optim.SGD(
                self.param_groups,
                lr=self.stage_config.get('learning_rate', self.config.learning_rate),
                momentum=0.9,
                weight_decay=self.stage_config.get('weight_decay', self.config.weight_decay),
                nesterov=True
            )
        elif optimizer_type == OptimizerType.ADAFACTOR:
            try:
                from transformers import Adafactor
                return Adafactor(
                    self.param_groups,
                    lr=self.stage_config.get('learning_rate', self.config.learning_rate),
                    scale_parameter=self.config.adafactor_scale_parameter,
                    relative_step_size=self.config.adafactor_relative_step,
                    warmup_init=True
                )
            except ImportError:
                warnings.warn("Adafactor not available, falling back to AdamW")
                return optim.AdamW(self.param_groups)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def _create_scheduler(self) -> Optional[_LRScheduler]:
        if self.total_steps is None:
            return None
        scheduler_type = self.stage_config.get('scheduler_type', self.config.scheduler_type)
        warmup_steps = self.stage_config.get('warmup_steps', self.config.warmup_steps)
        min_lr_ratio = self.stage_config.get('min_lr_ratio', self.config.min_lr_ratio)
        if scheduler_type == SchedulerType.COSINE:
            return CosineAnnealingLRWithWarmup(self.optimizer, warmup_steps, self.total_steps, min_lr_ratio=min_lr_ratio)
        elif scheduler_type == SchedulerType.COSINE_RESTART:
            restart_period = self.total_steps // self.config.num_cycles
            return CosineAnnealingWarmRestarts(self.optimizer, T_0=restart_period, T_mult=1,
                                              eta_min=self.config.learning_rate * min_lr_ratio)
        elif scheduler_type == SchedulerType.LINEAR:
            return LinearLRWithWarmup(self.optimizer, warmup_steps, self.total_steps, min_lr_ratio=min_lr_ratio)
        elif scheduler_type == SchedulerType.POLYNOMIAL:
            return PolynomialLRWithWarmup(self.optimizer, warmup_steps, self.total_steps, power=0.5, min_lr_ratio=min_lr_ratio)
        elif scheduler_type == SchedulerType.CUSTOM_MEDICAL:
            return MedicalLRScheduler(self.optimizer, warmup_steps, self.total_steps, stage=self.stage)
        else:
            return None

    def step(self, closure: Optional[Callable] = None):
        if self.sam_optimizer:
            self.sam_optimizer.step(closure)
        elif self.lookahead_optimizer:
            self.lookahead_optimizer.step()
        else:
            self.optimizer.step(closure)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_lr(self) -> List[float]:
        if self.scheduler:
            return self.scheduler.get_last_lr()
        else:
            return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self) -> Dict[str, Any]:
        state = {
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'stage': self.stage
        }
        if self.scheduler:
            state['scheduler'] = self.scheduler.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]):
        self.optimizer.load_state_dict(state['optimizer'])
        if 'scheduler' in state and self.scheduler:
            self.scheduler.load_state_dict(state['scheduler'])


class CosineAnnealingLRWithWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.01, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        return [base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))) for base_lr in self.base_lrs]


class LinearLRWithWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.01, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        return [base_lr * (1 - progress * (1 - self.min_lr_ratio)) for base_lr in self.base_lrs]


class PolynomialLRWithWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, power=1.0, min_lr_ratio=0.01, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.power = power
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        return [base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * (1 - progress) ** self.power) for base_lr in self.base_lrs]


class MedicalLRScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, stage, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.stage = stage
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            warmup_factor = self.last_epoch / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        if self.stage == 1:
            decay_factor = 0.5 * (1 + math.cos(math.pi * progress * 0.5))
            min_lr_ratio = 0.1
        elif self.stage == 2:
            cycle_progress = (progress * 2) % 1.0
            decay_factor = 0.5 * (1 + math.cos(math.pi * cycle_progress))
            min_lr_ratio = 0.05
        else:
            decay_factor = (1 - progress) ** 2
            min_lr_ratio = 0.01
        return [base_lr * (min_lr_ratio + (1 - min_lr_ratio) * decay_factor) for base_lr in self.base_lrs]


class SAM(torch.optim.Optimizer):
    def __init__(self, base_optimizer: torch.optim.Optimizer, rho: float = 0.05):
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        assert closure is not None, "SAM requires closure for second forward pass"
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(dtype=torch.float32).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            dtype=torch.float32
        )
        return norm

    def zero_grad(self):
        self.base_optimizer.zero_grad()


class Lookahead(torch.optim.Optimizer):
    def __init__(self, base_optimizer: torch.optim.Optimizer, k: int = 5, alpha: float = 0.5):
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state
        self.step_count = 0
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["slow_weights"] = p.data.clone()

    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        self.step_count += 1
        if self.step_count % self.k == 0:
            for group in self.param_groups:
                for p in group["params"]:
                    slow = self.state[p]["slow_weights"]
                    slow.add_(p.data - slow, alpha=self.alpha)
                    p.data.copy_(slow)
        return loss

    def zero_grad(self):
        self.base_optimizer.zero_grad()


def create_optimizer(model: torch.nn.Module, stage: int, total_steps: Optional[int] = None,
                     config: Optional[OptimizationConfig] = None, **kwargs) -> MedicalOptimizer:
    if config is None:
        config = OptimizationConfig(**kwargs)
    return MedicalOptimizer(model, config, stage, total_steps)


def create_scheduler(optimizer: torch.optim.Optimizer, scheduler_type: SchedulerType, total_steps: int,
                     warmup_steps: int = 1000, **kwargs) -> _LRScheduler:
    if scheduler_type == SchedulerType.COSINE:
        return CosineAnnealingLRWithWarmup(optimizer, warmup_steps, total_steps, **kwargs)
    elif scheduler_type == SchedulerType.LINEAR:
        return LinearLRWithWarmup(optimizer, warmup_steps, total_steps, **kwargs)
    elif scheduler_type == SchedulerType.POLYNOMIAL:
        return PolynomialLRWithWarmup(optimizer, warmup_steps, total_steps, **kwargs)
    elif scheduler_type == SchedulerType.CUSTOM_MEDICAL:
        return MedicalLRScheduler(optimizer, warmup_steps, total_steps, **kwargs)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def clip_gradients(model: torch.nn.Module, max_norm: float, norm_type: float = 2.0) -> float:
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type)


def get_gradient_norm(model: torch.nn.Module, norm_type: float = 2.0) -> float:
    parameters = [p for p in model.parameters() if p.grad is not None]
    if len(parameters) == 0:
        return 0.0
    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
        norm_type
    )
    return total_norm.item()


if __name__ == "__main__":
    print("Testing Medical Optimization Framework...")

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_encoder = torch.nn.Linear(1000, 512)
            self.hcf_fusion = torch.nn.Linear(512, 512)
            self.llama_layers = torch.nn.Linear(512, 32000)
            self.lora_a = torch.nn.Linear(512, 16)
            self.lora_b = torch.nn.Linear(16, 512)

    model = DummyModel()
    total_steps = 10000

    for stage in [1, 2, 3]:
        print(f"\n=== Testing Stage {stage} ===")
        optimizer = create_optimizer(
            model=model,
            stage=stage,
            total_steps=total_steps,
            learning_rate=1e-4,
            use_sam=False,
            use_lookahead=False
        )
        print(f"Base optimizer: {type(optimizer.optimizer).__name__}")
        print(f"Scheduler: {type(optimizer.scheduler).__name__}")
        print(f"Parameter groups: {len(optimizer.param_groups)}")
        for i, group in enumerate(optimizer.param_groups):
            print(f"  Group {i} ({group['name']}): LR = {group['lr']:.2e}")
        lrs = []
        for step in [0, 1000, 5000, 9000, 10000]:
            if optimizer.scheduler:
                optimizer.scheduler.last_epoch = step
                current_lrs = optimizer.scheduler.get_lr()
                lrs.append((step, current_lrs[0]))
        print(f"LR schedule: {lrs}")

    print("\nOptimization testing completed successfully!")
