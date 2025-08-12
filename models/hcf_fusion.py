import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math
from einops import rearrange, repeat
from dataclasses import dataclass


@dataclass
class HCFConfig:
    vision_dim: int = 1024
    text_dim: int = 4096
    fusion_dim: int = 4096
    
    num_levels: int = 3
    num_heads: int = 16
    
    num_fusion_layers: int = 3
    dropout: float = 0.1
    
    use_adaptive_gating: bool = True
    gate_activation: str = "sigmoid"
    
    use_cross_attention: bool = True
    attention_dropout: float = 0.1
    
    norm_type: str = "layer_norm"
    
    use_position_encoding: bool = True
    use_temperature_scaling: bool = True
    temperature_init: float = 1.0


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class AdaptiveGatingMechanism(nn.Module):
    
    def __init__(self, config: HCFConfig):
        super().__init__()
        self.config = config
        
        self.vision_gate = nn.Sequential(
            nn.Linear(config.vision_dim, config.fusion_dim // 4),
            nn.ReLU(),
            nn.Linear(config.fusion_dim // 4, 1)
        )
        
        self.text_gate = nn.Sequential(
            nn.Linear(config.text_dim, config.fusion_dim // 4),
            nn.ReLU(),
            nn.Linear(config.fusion_dim // 4, 1)
        )
        
        self.fusion_gate = nn.Sequential(
            nn.Linear(config.fusion_dim, config.fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_dim // 2, config.num_levels)
        )
        
        if config.gate_activation == "sigmoid":
            self.activation = torch.sigmoid
        elif config.gate_activation == "tanh":
            self.activation = torch.tanh
        else:
            self.activation = F.gelu
    
    def forward(
        self, 
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        fused_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v_gate = self.activation(self.vision_gate(vision_features))
        t_gate = self.activation(self.text_gate(text_features))
        
        level_gates = F.softmax(self.fusion_gate(fused_features), dim=-1)
        
        return v_gate, t_gate, level_gates


class CrossModalAttention(nn.Module):
    
    def __init__(self, config: HCFConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.fusion_dim // config.num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(config.fusion_dim, config.fusion_dim)
        self.k_proj = nn.Linear(config.fusion_dim, config.fusion_dim) 
        self.v_proj = nn.Linear(config.fusion_dim, config.fusion_dim)
        self.out_proj = nn.Linear(config.fusion_dim, config.fusion_dim)
        
        self.dropout = nn.Dropout(config.attention_dropout)
        
        if config.use_temperature_scaling:
            self.temperature = nn.Parameter(torch.tensor(config.temperature_init))
        else:
            self.temperature = 1.0
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor, 
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale / self.temperature
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.config.fusion_dim
        )
        output = self.out_proj(output)
        
        attn_weights = attn_weights.mean(dim=1)
        
        return output, attn_weights


class HierarchicalFusionLayer(nn.Module):
    
    def __init__(self, config: HCFConfig, level: int):
        super().__init__()
        self.config = config
        self.level = level
        
        if level == 0:
            self.feature_transform = nn.Sequential(
                nn.Linear(config.fusion_dim, config.fusion_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.fusion_dim, config.fusion_dim)
            )
        elif level == 1:
            self.feature_transform = nn.Sequential(
                nn.Linear(config.fusion_dim, config.fusion_dim * 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.fusion_dim * 2, config.fusion_dim)
            )
        else:
            self.feature_transform = nn.Sequential(
                nn.Linear(config.fusion_dim, config.fusion_dim * 4),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.fusion_dim * 4, config.fusion_dim * 2),
                nn.GELU(),
                nn.Linear(config.fusion_dim * 2, config.fusion_dim)
            )
        
        if config.use_cross_attention:
            self.cross_attention = CrossModalAttention(config)
        
        if config.norm_type == "layer_norm":
            self.norm1 = nn.LayerNorm(config.fusion_dim)
            self.norm2 = nn.LayerNorm(config.fusion_dim)
        elif config.norm_type == "batch_norm":
            self.norm1 = nn.BatchNorm1d(config.fusion_dim)
            self.norm2 = nn.BatchNorm1d(config.fusion_dim)
        
        self.fusion_weights = nn.Parameter(torch.ones(2) * 0.5)
        
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vision_norm = self.norm1(vision_features)
        text_norm = self.norm1(text_features)
        
        vision_transformed = self.feature_transform(vision_norm)
        text_transformed = self.feature_transform(text_norm)
        
        if self.config.use_cross_attention:
            v2t_output, v2t_weights = self.cross_attention(
                vision_transformed, text_transformed, text_transformed, attention_mask
            )
            
            t2v_output, t2v_weights = self.cross_attention(
                text_transformed, vision_transformed, vision_transformed
            )
            
            vision_attended = vision_transformed + v2t_output
            text_attended = text_transformed + t2v_output
            
            attention_weights = {
                'v2t': v2t_weights,
                't2v': t2v_weights
            }
        else:
            vision_attended = vision_transformed
            text_attended = text_transformed
            attention_weights = None
        
        fusion_weights = F.softmax(self.fusion_weights, dim=0)
        
        if self.level == 0:
            fused = fusion_weights[0] * vision_attended + fusion_weights[1] * text_attended
        elif self.level == 1:
            fused = (fusion_weights[0] * 1.2) * vision_attended + fusion_weights[1] * text_attended
        else:
            fused = fusion_weights[0] * vision_attended + (fusion_weights[1] * 1.2) * text_attended
        
        fused = self.norm2(fused + vision_features[:, :fused.shape[1]] if vision_features.shape[1] >= fused.shape[1] 
                          else fused + text_features[:, :fused.shape[1]])
        
        return fused, attention_weights


class HierarchicalCrossModalFusion(nn.Module):
    
    def __init__(self, config: HCFConfig = None, **kwargs):
        super().__init__()
        
        if config is None:
            config = HCFConfig(**kwargs)
        self.config = config
        
        self.vision_proj = nn.Linear(config.vision_dim, config.fusion_dim)
        self.text_proj = nn.Linear(config.text_dim, config.fusion_dim)
        
        if config.use_position_encoding:
            self.pos_encoding = PositionalEncoding(config.fusion_dim)
        
        self.fusion_layers = nn.ModuleList([
            HierarchicalFusionLayer(config, level=i) 
            for i in range(config.num_levels)
        ])
        
        if config.use_adaptive_gating:
            self.adaptive_gating = AdaptiveGatingMechanism(config)
        
        self.level_combiner = nn.Sequential(
            nn.Linear(config.fusion_dim * config.num_levels, config.fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_dim * 2, config.fusion_dim)
        )
        
        self.text_output_proj = nn.Linear(config.fusion_dim, config.text_dim)
        self.vision_output_proj = nn.Linear(config.fusion_dim, config.vision_dim)
        
        self.clinical_validator = nn.Sequential(
            nn.Linear(config.fusion_dim, config.fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_intermediate: bool = False
    ) -> Union[Tuple[torch.Tensor, Dict], torch.Tensor]:
        batch_size = vision_features.shape[0]
        
        if len(vision_features.shape) == 2:
            vision_features = vision_features.unsqueeze(1)
        
        vision_proj = self.vision_proj(vision_features)
        text_proj = self.text_proj(text_features)
        
        if self.config.use_position_encoding:
            vision_proj = self.pos_encoding(vision_proj.transpose(0, 1)).transpose(0, 1)
            text_proj = self.pos_encoding(text_proj.transpose(0, 1)).transpose(0, 1)
        
        intermediate_results = {
            'level_outputs': [],
            'attention_weights': [],
            'gating_weights': []
        }
        
        level_outputs = []
        
        for level, fusion_layer in enumerate(self.fusion_layers):
            level_output, attn_weights = fusion_layer(
                vision_proj, text_proj, attention_mask
            )
            
            level_outputs.append(level_output)
            
            if return_intermediate:
                intermediate_results['level_outputs'].append(level_output)
                intermediate_results['attention_weights'].append(attn_weights)
            
            if level < len(self.fusion_layers) - 1:
                alpha = 0.7 ** (level + 1)
                vision_proj = alpha * vision_proj + (1 - alpha) * level_output[:, :vision_proj.shape[1]]
                text_proj = alpha * text_proj + (1 - alpha) * level_output[:, :text_proj.shape[1]]
        
        max_seq_len = max(output.shape[1] for output in level_outputs)
        padded_outputs = []
        
        for output in level_outputs:
            if output.shape[1] < max_seq_len:
                padding = torch.zeros(
                    batch_size, max_seq_len - output.shape[1], self.config.fusion_dim,
                    device=output.device, dtype=output.dtype
                )
                output = torch.cat([output, padding], dim=1)
            padded_outputs.append(output)
        
        combined_levels = torch.cat(padded_outputs, dim=-1)
        
        fused_features = self.level_combiner(combined_levels)
        
        if self.config.use_adaptive_gating and hasattr(self, 'adaptive_gating'):
            vision_mean = vision_features.mean(dim=1) if len(vision_features.shape) > 2 else vision_features
            text_mean = text_features.mean(dim=1)
            fused_mean = fused_features.mean(dim=1)
            
            v_gate, t_gate, level_gates = self.adaptive_gating(
                vision_mean, text_mean, fused_mean
            )
            
            fused_features = fused_features * level_gates.unsqueeze(1)
            
            if return_intermediate:
                intermediate_results['gating_weights'] = {
                    'vision_gate': v_gate,
                    'text_gate': t_gate,
                    'level_gates': level_gates
                }
        
        clinical_score = self.clinical_validator(fused_features.mean(dim=1))
        
        if return_intermediate:
            intermediate_results.update({
                'clinical_score': clinical_score,
                'vision_projected': vision_proj,
                'text_projected': text_proj,
                'final_fused': fused_features
            })
            return fused_features, intermediate_results
        
        return fused_features
    
    def vision_only_forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        batch_size = vision_features.shape[0]
        
        if len(vision_features.shape) == 2:
            vision_features = vision_features.unsqueeze(1)
        
        vision_proj = self.vision_proj(vision_features)
        
        dummy_text = torch.zeros(
            batch_size, 1, self.config.fusion_dim,
            device=vision_features.device,
            dtype=vision_features.dtype
        )
        
        fused_features, _ = self.fusion_layers[0](vision_proj, dummy_text)
        
        return fused_features
    
    def get_attention_maps(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        _, intermediate = self.forward(
            vision_features, text_features, attention_mask, return_intermediate=True
        )
        
        attention_maps = {}
        for level, attn_weights in enumerate(intermediate['attention_weights']):
            if attn_weights is not None:
                attention_maps[f'level_{level}_v2t'] = attn_weights['v2t']
                attention_maps[f'level_{level}_t2v'] = attn_weights['t2v']
        
        return attention_maps
    
    def compute_fusion_loss(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        fused_features, intermediate = self.forward(
            vision_features, text_features, attention_mask, return_intermediate=True
        )
        
        losses = {}
        
        clinical_score = intermediate['clinical_score']
        losses['clinical_loss'] = F.mse_loss(clinical_score, torch.ones_like(clinical_score))
        
        level_outputs = intermediate['level_outputs']
        if len(level_outputs) > 1:
            diversity_loss = 0.0
            for i in range(len(level_outputs)):
                for j in range(i + 1, len(level_outputs)):
                    sim = F.cosine_similarity(
                        level_outputs[i].mean(dim=1), 
                        level_outputs[j].mean(dim=1), 
                        dim=1
                    ).mean()
                    diversity_loss += sim
            
            losses['diversity_loss'] = diversity_loss / (len(level_outputs) * (len(level_outputs) - 1) / 2)
        
        if 'gating_weights' in intermediate:
            gating = intermediate['gating_weights']
            if 'level_gates' in gating:
                level_gates = gating['level_gates']
                uniform_target = torch.ones_like(level_gates) / level_gates.shape[-1]
                losses['gating_loss'] = F.kl_div(
                    F.log_softmax(level_gates, dim=-1),
                    uniform_target,
                    reduction='batchmean'
                )
        
        return losses


def create_hcf_module(
    vision_dim: int = 1024,
    text_dim: int = 4096,
    fusion_dim: int = 4096,
    num_levels: int = 3,
    **kwargs
) -> HierarchicalCrossModalFusion:
    config = HCFConfig(
        vision_dim=vision_dim,
        text_dim=text_dim,
        fusion_dim=fusion_dim,
        num_levels=num_levels,
        **kwargs
    )
    
    return HierarchicalCrossModalFusion(config)


if __name__ == "__main__":
    print("Testing Hierarchical Cross-Modal Fusion module...")
    
    hcf = create_hcf_module(
        vision_dim=1024,
        text_dim=4096,
        fusion_dim=2048,
        num_levels=3,
        use_adaptive_gating=True,
        use_cross_attention=True
    )
    
    batch_size = 4
    vision_seq_len = 196
    text_seq_len = 128
    
    vision_features = torch.randn(batch_size, vision_seq_len, 1024)
    text_features = torch.randn(batch_size, text_seq_len, 4096)
    attention_mask = torch.ones(batch_size, text_seq_len)
    
    print(f"Input shapes - Vision: {vision_features.shape}, Text: {text_features.shape}")
    
    fused_output = hcf(vision_features, text_features, attention_mask)
    print(f"Fused output shape: {fused_output.shape}")
    
    fused_output, intermediate = hcf(
        vision_features, text_features, attention_mask, return_intermediate=True
    )
    
    print(f"Number of level outputs: {len(intermediate['level_outputs'])}")
    print(f"Clinical score shape: {intermediate['clinical_score'].shape}")
    
    attention_maps = hcf.get_attention_maps(vision_features, text_features, attention_mask)
    print(f"Attention maps keys: {list(attention_maps.keys())}")
    
    fusion_losses = hcf.compute_fusion_loss(vision_features, text_features, attention_mask)
    print(f"Fusion losses: {list(fusion_losses.keys())}")
    
    vision_only_output = hcf.vision_only_forward(vision_features)
    print(f"Vision-only output shape: {vision_only_output.shape}")
    
    print("HCF module testing completed successfully!")
