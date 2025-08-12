from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import yaml
import json

@dataclass 
class StageConfig:
    stage: int
    batch_size: int = 8
    num_epochs: int = 10
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    lora_rank: int = 16
    lora_alpha: int = 32
    eval_every_n_epochs: int = 2
    save_every_n_epochs: int = 5
    datasets: List[str] = None
    
    def __post_init__(self):
        if self.datasets is None:
            if self.stage == 1:
                self.datasets = ["roco_v2"]
            elif self.stage == 2:
                self.datasets = ["roco_v2", "padchest"]
            else:
                self.datasets = ["roco_v2", "padchest"]

@dataclass
class TrainingConfig:
    padchest_root: str
    roco_root: str
    image_size: tuple = (512, 512)
    max_text_length: int = 512
    
    model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    vision_backbone: str = "efficientnet_b7"
    fusion_layers: int = 3
    
    output_dir: str = "./radixpert_training"
    stages_to_run: List[int] = None
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    
    num_workers: int = 4
    log_level: str = "INFO"
    
    wandb_project: str = "radixpert"
    wandb_entity: Optional[str] = None
    
    def __post_init__(self):
        if self.stages_to_run is None:
            self.stages_to_run = [1, 2, 3]

def create_stage_configs(
    batch_sizes: List[int] = [16, 12, 8],
    num_epochs: List[int] = [10, 15, 20],
    learning_rates: List[float] = [5e-5, 3e-5, 1e-5],
    lora_ranks: List[int] = [8, 12, 16]
) -> Dict[int, StageConfig]:
    
    stage_configs = {}
    
    for i, stage in enumerate([1, 2, 3]):
        stage_configs[stage] = StageConfig(
            stage=stage,
            batch_size=batch_sizes[i] if i < len(batch_sizes) else 8,
            num_epochs=num_epochs[i] if i < len(num_epochs) else 10,
            learning_rate=learning_rates[i] if i < len(learning_rates) else 1e-5,
            lora_rank=lora_ranks[i] if i < len(lora_ranks) else 16
        )
    
    return stage_configs

def save_config_to_file(config: TrainingConfig, file_path: str):
    config_dict = {
        'data': {
            'padchest_root': config.padchest_root,
            'roco_root': config.roco_root,
            'image_size': config.image_size,
            'max_text_length': config.max_text_length
        },
        'model': {
            'model_name': config.model_name,
            'vision_backbone': config.vision_backbone,
            'fusion_layers': config.fusion_layers
        },
        'training': {
            'output_dir': config.output_dir,
            'stages_to_run': config.stages_to_run,
            'mixed_precision': config.mixed_precision,
            'gradient_checkpointing': config.gradient_checkpointing
        },
        'system': {
            'num_workers': config.num_workers,
            'log_level': config.log_level
        },
        'experiment': {
            'wandb_project': config.wandb_project,
            'wandb_entity': config.wandb_entity
        }
    }
    
    with open(file_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
