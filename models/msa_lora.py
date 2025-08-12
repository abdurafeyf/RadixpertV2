import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union
import math
from dataclasses import dataclass
from enum import Enum


class TrainingStage(Enum):
    PRE_ALIGNMENT = 1
    DOMAIN_ADAPTATION = 2
    TASK_SPECIFIC = 3


@dataclass
class MSALoRAConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1
    
    stage: int = 1
    target_modules: List[str] = None
    
    stage_scaling: Dict[int, float] = None
    adaptive_rank: bool = True
    rank_schedule: Dict[int, int] = None
    
    l2_reg: float = 1e-5
    orthogonal_reg: float = 1e-4
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        
        if self.stage_scaling is None:
            self.stage_scaling = {
                1: 0.5,
                2: 0.8,
                3: 1.0
            }
        
        if self.rank_schedule is None:
            self.rank_schedule = {
                1: max(4, self.rank // 4),
                2: max(8, self.rank // 2),
                3: self.rank
            }


class MSALoRALayer(nn.Module):
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int,
        config: MSALoRAConfig,
        original_layer: nn.Linear
    ):
        super().__init__()
        self.config = config
        self.in_features = in_features
        self.out_features = out_features
        
        self.original_layer = original_layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
            
        self.current_rank = self._get_stage_rank()
        
        self.lora_A = nn.Parameter(torch.randn(self.current_rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.current_rank))
        
        self.scaling = self.config.alpha / self.current_rank * self.config.stage_scaling[config.stage]
        
        self.dropout = nn.Dropout(config.dropout)
        
        self._init_lora_weights()
        
        if config.stage >= 2:
            self.batch_norm = nn.BatchNorm1d(out_features, affine=False)
        else:
            self.batch_norm = None
            
        if config.stage == 3:
            self.adaptive_gate = nn.Parameter(torch.ones(1))
        else:
            self.adaptive_gate = None
    
    def _get_stage_rank(self) -> int:
        return self.config.rank_schedule.get(self.config.stage, self.config.rank)
    
    def _init_lora_weights(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_output = self.original_layer(x)
        
        lora_output = self._compute_lora_output(x)
        
        if self.config.stage == 1:
            output = original_output + lora_output * 0.1
        elif self.config.stage == 2:
            if self.batch_norm is not None:
                shape = lora_output.shape
                if len(shape) > 2:
                    lora_output = lora_output.view(-1, shape[-1])
                    lora_output = self.batch_norm(lora_output)
                    lora_output = lora_output.view(shape)
                else:
                    lora_output = self.batch_norm(lora_output)
            output = original_output + lora_output
        else:
            if self.adaptive_gate is not None:
                lora_output = lora_output * torch.sigmoid(self.adaptive_gate)
            output = original_output + lora_output
            
        return output
    
    def _compute_lora_output(self, x: torch.Tensor) -> torch.Tensor:
        x_drop = self.dropout(x)
        
        lora_output = x_drop @ self.lora_A.T
        lora_output = lora_output @ self.lora_B.T
        lora_output = lora_output * self.scaling
        
        return lora_output
    
    def get_lora_parameters(self) -> Dict[str, torch.Tensor]:
        params = {
            'lora_A': self.lora_A,
            'lora_B': self.lora_B,
            'scaling': self.scaling,
            'rank': self.current_rank
        }
        
        if self.adaptive_gate is not None:
            params['adaptive_gate'] = self.adaptive_gate
            
        return params
    
    def compute_regularization_loss(self) -> torch.Tensor:
        reg_loss = 0.0
        
        if self.config.l2_reg > 0:
            reg_loss += self.config.l2_reg * (
                torch.norm(self.lora_A, p=2) + torch.norm(self.lora_B, p=2)
            )
        
        if self.config.orthogonal_reg > 0 and self.current_rank > 1:
            A_orth = self.lora_A @ self.lora_A.T
            A_identity = torch.eye(self.current_rank, device=A_orth.device)
            reg_loss += self.config.orthogonal_reg * torch.norm(A_orth - A_identity, p='fro')
            
            B_orth = self.lora_B @ self.lora_B.T
            B_identity = torch.eye(self.out_features, device=B_orth.device)
            if self.out_features <= self.current_rank * 4:
                reg_loss += self.config.orthogonal_reg * torch.norm(B_orth - B_identity, p='fro')
        
        return reg_loss
    
    def update_stage(self, new_stage: int):
        if new_stage == self.config.stage:
            return
            
        old_stage = self.config.stage
        self.config.stage = new_stage
        
        new_rank = self._get_stage_rank()
        if new_rank != self.current_rank:
            self._resize_lora_matrices(new_rank)
        
        self.scaling = self.config.alpha / self.current_rank * self.config.stage_scaling[new_stage]
        
        if new_stage >= 2 and old_stage < 2:
            self.batch_norm = nn.BatchNorm1d(self.out_features, affine=False)
            self.batch_norm = self.batch_norm.to(self.lora_A.device)
            
        if new_stage == 3 and old_stage < 3:
            self.adaptive_gate = nn.Parameter(torch.ones(1, device=self.lora_A.device))
            
        print(f"Updated LoRA layer from Stage {old_stage} to Stage {new_stage}")
        print(f"Rank: {self.current_rank}, Scaling: {self.scaling:.4f}")
    
    def _resize_lora_matrices(self, new_rank: int):
        old_rank = self.current_rank
        
        if new_rank > old_rank:
            new_A = torch.zeros(new_rank, self.in_features, device=self.lora_A.device)
            new_B = torch.zeros(self.out_features, new_rank, device=self.lora_B.device)
            
            new_A[:old_rank] = self.lora_A.data
            new_B[:, :old_rank] = self.lora_B.data
            
            with torch.no_grad():
                nn.init.kaiming_uniform_(new_A[old_rank:], a=math.sqrt(5))
                
        else:
            with torch.no_grad():
                W_lora = self.lora_B @ self.lora_A
                U, S, Vh = torch.svd(W_lora)
                
                new_A = Vh[:new_rank]
                new_B = U[:, :new_rank] @ torch.diag(S[:new_rank])
        
        self.lora_A = nn.Parameter(new_A)
        self.lora_B = nn.Parameter(new_B)
        self.current_rank = new_rank


class MSALoRAManager:
    
    def __init__(self, model: nn.Module, config: MSALoRAConfig):
        self.model = model
        self.config = config
        self.lora_layers: Dict[str, MSALoRALayer] = {}
        
        self._apply_lora()
    
    def _apply_lora(self):
        for name, module in self.model.named_modules():
            if self._is_target_module(name, module):
                lora_layer = MSALoRALayer(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    config=self.config,
                    original_layer=module
                )
                
                self._replace_module(name, lora_layer)
                self.lora_layers[name] = lora_layer
                
        print(f"Applied MSA-LoRA to {len(self.lora_layers)} modules")
    
    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        return (
            isinstance(module, nn.Linear) and 
            any(target in name for target in self.config.target_modules)
        )
    
    def _replace_module(self, name: str, new_module: nn.Module):
        names = name.split('.')
        parent = self.model
        for name_part in names[:-1]:
            parent = getattr(parent, name_part)
        setattr(parent, names[-1], new_module)
    
    def update_stage(self, new_stage: int):
        print(f"Updating all MSA-LoRA layers to Stage {new_stage}")
        for name, lora_layer in self.lora_layers.items():
            lora_layer.update_stage(new_stage)
    
    def get_trainable_parameters(self) -> Dict[str, int]:
        total = 0
        by_layer = {}
        
        for name, lora_layer in self.lora_layers.items():
            layer_params = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
            by_layer[name] = layer_params
            total += layer_params
            
        return {'total': total, 'by_layer': by_layer}
    
    def compute_regularization_loss(self) -> torch.Tensor:
        total_reg_loss = 0.0
        
        for lora_layer in self.lora_layers.values():
            total_reg_loss += lora_layer.compute_regularization_loss()
            
        return total_reg_loss
    
    def save_lora_state(self, path: str):
        state = {
            'config': self.config,
            'lora_layers': {
                name: {
                    'state_dict': layer.state_dict(),
                    'lora_params': layer.get_lora_parameters()
                }
                for name, layer in self.lora_layers.items()
            }
        }
        torch.save(state, path)
        
    def load_lora_state(self, path: str):
        state = torch.load(path, map_location='cpu')
        
        for name, layer_state in state['lora_layers'].items():
            if name in self.lora_layers:
                self.lora_layers[name].load_state_dict(layer_state['state_dict'])


def create_msa_lora_model(
    base_model: nn.Module,
    stage: int = 1,
    rank: int = 16,
    alpha: int = 32,
    target_modules: List[str] = None,
    **kwargs
) -> MSALoRAManager:
    config = MSALoRAConfig(
        stage=stage,
        rank=rank,
        alpha=alpha,
        target_modules=target_modules,
        **kwargs
    )
    
    return MSALoRAManager(base_model, config)


if __name__ == "__main__":
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(512, 512)
            self.v_proj = nn.Linear(512, 512) 
            self.k_proj = nn.Linear(512, 512)
            self.output = nn.Linear(512, 100)
            
        def forward(self, x):
            q = self.q_proj(x)
            v = self.v_proj(x)
            k = self.k_proj(x)
            return self.output(q + v + k)
    
    model = TestModel()
    print(f"Original model parameters: {sum(p.numel() for p in model.parameters())}")
    
    lora_manager = create_msa_lora_model(
        model, 
        stage=1, 
        rank=16,
        target_modules=["q_proj", "v_proj", "k_proj"]
    )
    
    trainable_params = lora_manager.get_trainable_parameters()
    print(f"Trainable parameters after LoRA: {trainable_params}")
    
    x = torch.randn(32, 512)
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    lora_manager.update_stage(2)
    lora_manager.update_stage(3)
    
    final_params = lora_manager.get_trainable_parameters()
    print(f"Final trainable parameters: {final_params}")
