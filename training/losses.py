import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from dataclasses import dataclass
import math
from enum import Enum


class LossType(Enum):
    GENERATION = "generation"
    CONTRASTIVE = "contrastive"
    CLINICAL = "clinical"
    FUSION = "fusion"
    REGULARIZATION = "regularization"
    MSA_LORA = "msa_lora"


@dataclass
class LossConfig:
    stage_weights: Dict[int, Dict[str, float]] = None
    label_smoothing: float = 0.1
    ignore_index: int = -100
    temperature: float = 0.07
    margin: float = 0.2
    clinical_weight: float = 0.5
    clinical_threshold: float = 0.8
    diversity_weight: float = 0.1
    attention_entropy_weight: float = 0.05
    l2_weight: float = 1e-5
    orthogonal_weight: float = 1e-4
    sparsity_weight: float = 1e-6
    use_focal_loss: bool = False
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    def __post_init__(self):
        if self.stage_weights is None:
            self.stage_weights = {
                1: {
                    "generation": 1.0,
                    "contrastive": 0.5,
                    "clinical": 0.0,
                    "fusion": 0.0,
                    "regularization": 0.1,
                    "msa_lora": 0.1
                },
                2: {
                    "generation": 1.0,
                    "contrastive": 0.7,
                    "clinical": 0.3,
                    "fusion": 0.2,
                    "regularization": 0.1,
                    "msa_lora": 0.15
                },
                3: {
                    "generation": 1.0,
                    "contrastive": 0.8,
                    "clinical": 0.5,
                    "fusion": 0.3,
                    "regularization": 0.1,
                    "msa_lora": 0.2
                }
            }


class GenerationLoss(nn.Module):
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config
        if config.use_focal_loss:
            self.loss_fn = FocalLoss(
                alpha=config.focal_alpha,
                gamma=config.focal_gamma,
                ignore_index=config.ignore_index
            )
        else:
            self.loss_fn = nn.CrossEntropyLoss(
                label_smoothing=config.label_smoothing,
                ignore_index=config.ignore_index
            )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[
        str, torch.Tensor]:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        loss = self.loss_fn(flat_logits, flat_labels)
        with torch.no_grad():
            perplexity = torch.exp(loss.clamp(max=10))
        with torch.no_grad():
            predictions = flat_logits.argmax(dim=-1)
            mask = (flat_labels != self.config.ignore_index)
            accuracy = (predictions == flat_labels)[mask].float().mean()
        return {
            'loss': loss,
            'perplexity': perplexity,
            'accuracy': accuracy,
            'num_tokens': mask.sum() if 'mask' in locals() else torch.tensor(flat_labels.numel())
        }


class ContrastiveLoss(nn.Module):
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config
        self.temperature = config.temperature
        self.margin = config.margin

    def forward(self, vision_features: torch.Tensor, text_features: torch.Tensor, labels: Optional[
        torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size = vision_features.size(0)
        vision_features = F.normalize(vision_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        similarity_matrix = torch.matmul(vision_features, text_features.t()) / self.temperature
        if labels is None:
            labels = torch.arange(batch_size, device=vision_features.device)
        v2t_loss = F.cross_entropy(similarity_matrix, labels)
        t2v_loss = F.cross_entropy(similarity_matrix.t(), labels)
        contrastive_loss = (v2t_loss + t2v_loss) / 2
        with torch.no_grad():
            v2t_acc = (similarity_matrix.argmax(dim=1) == labels).float().mean()
            t2v_acc = (similarity_matrix.t().argmax(dim=1) == labels).float().mean()
            alignment_acc = (v2t_acc + t2v_acc) / 2
        with torch.no_grad():
            positive_sim = similarity_matrix.diag().mean()
            negative_sim = (similarity_matrix.sum() - similarity_matrix.diag().sum()) / (
                        batch_size * (batch_size - 1))
        return {
            'loss': contrastive_loss,
            'v2t_loss': v2t_loss,
            't2v_loss': t2v_loss,
            'alignment_accuracy': alignment_acc,
            'positive_similarity': positive_sim,
            'negative_similarity': negative_sim
        }


class ClinicalValidationLoss(nn.Module):
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config
        self.threshold = config.clinical_threshold
        self.clinical_classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, fused_features: torch.Tensor, clinical_labels: Optional[torch.Tensor] = None,
                generated_text: Optional[List[str]] = None, reference_text: Optional[List[str]] = None) -> Dict[
        str, torch.Tensor]:
        batch_size = fused_features.size(0)
        if len(fused_features.shape) > 2:
            pooled_features = fused_features.mean(dim=1)
        else:
            pooled_features = fused_features
        clinical_scores = self.clinical_classifier(pooled_features).squeeze(-1)
        if clinical_labels is not None:
            clinical_loss = F.binary_cross_entropy(clinical_scores, clinical_labels.float())
        else:
            if generated_text and reference_text:
                heuristic_scores = self._compute_heuristic_scores(generated_text, reference_text)
                heuristic_scores = torch.tensor(heuristic_scores, device=clinical_scores.device)
                clinical_loss = F.mse_loss(clinical_scores, heuristic_scores)
            else:
                target_scores = torch.ones_like(clinical_scores) * 0.8
                clinical_loss = F.mse_loss(clinical_scores, target_scores)
        with torch.no_grad():
            predicted_valid = (clinical_scores > self.threshold).float()
            if clinical_labels is not None:
                true_valid = (clinical_labels > self.threshold).float()
                clinical_accuracy = (predicted_valid == true_valid).float().mean()
            else:
                clinical_accuracy = predicted_valid.mean()
        return {
            'loss': clinical_loss,
            'clinical_scores': clinical_scores,
            'clinical_accuracy': clinical_accuracy,
            'mean_clinical_score': clinical_scores.mean()
        }

    def _compute_heuristic_scores(self, generated_text: List[str], reference_text: List[str]) -> List[float]:
        scores = []
        for gen_text, ref_text in zip(generated_text, reference_text):
            score = 0.0
            medical_terms = [
                'normal', 'abnormal', 'findings', 'impression', 'chest', 'lung', 'heart',
                'radiograph', 'patient', 'examination', 'study', 'image', 'anatomy'
            ]
            gen_lower = gen_text.lower()
            term_score = sum(1 for term in medical_terms if term in gen_lower) / len(medical_terms)
            score += term_score * 0.3
            gen_len = len(gen_text.split())
            if 10 <= gen_len <= 200:
                length_score = 1.0
            else:
                length_score = max(0.0, 1.0 - abs(gen_len - 100) / 100)
            score += length_score * 0.2
            structure_indicators = ['findings:', 'impression:', 'technique:', 'comparison:']
            structure_score = min(1.0, sum(1 for ind in structure_indicators if ind in gen_lower) * 0.5)
            score += structure_score * 0.2
            common_words = set(gen_text.lower().split()) & set(ref_text.lower().split())
            semantic_score = min(1.0, len(common_words) / max(10, len(set(ref_text.lower().split()))))
            score += semantic_score * 0.3
            scores.append(min(1.0, max(0.0, score)))
        return scores


class FusionLoss(nn.Module):
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config

    def forward(self, fusion_outputs: Dict[str, torch.Tensor], attention_weights: Optional[Dict[str, torch.Tensor]] = None) -> Dict[
        str, torch.Tensor]:
        losses = {}
        total_loss = 0.0
        if 'level_outputs' in fusion_outputs:
            diversity_loss = self._compute_diversity_loss(fusion_outputs['level_outputs'])
            losses['diversity_loss'] = diversity_loss
            total_loss += diversity_loss * self.config.diversity_weight
        if attention_weights:
            entropy_loss = self._compute_attention_entropy_loss(attention_weights)
            losses['attention_entropy_loss'] = entropy_loss
            total_loss += entropy_loss * self.config.attention_entropy_weight
        if 'gating_weights' in fusion_outputs:
            gating_loss = self._compute_gating_regularization(fusion_outputs['gating_weights'])
            losses['gating_loss'] = gating_loss
            total_loss += gating_loss * 0.1
        losses['total_fusion_loss'] = total_loss
        return losses

    def _compute_diversity_loss(self, level_outputs: List[torch.Tensor]) -> torch.Tensor:
        if len(level_outputs) < 2:
            return torch.tensor(0.0, device=level_outputs[0].device)
        diversity_loss = 0.0
        num_pairs = 0
        for i in range(len(level_outputs)):
            for j in range(i + 1, len(level_outputs)):
                feat_i = level_outputs[i].mean(dim=1)
                feat_j = level_outputs[j].mean(dim=1)
                similarity = F.cosine_similarity(feat_i, feat_j, dim=1).mean()
                diversity_loss += similarity
                num_pairs += 1
        return diversity_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0)

    def _compute_attention_entropy_loss(self, attention_weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        entropy_loss = 0.0
        num_layers = 0
        for _, weights in attention_weights.items():
            if weights is not None and len(weights.shape) >= 2:
                probs = F.softmax(weights, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
                entropy_loss += entropy
                num_layers += 1
        return entropy_loss / num_layers if num_layers > 0 else torch.tensor(0.0)

    def _compute_gating_regularization(self, gating_weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        reg_loss = 0.0
        for gate_name, weights in gating_weights.items():
            if 'level_gates' in gate_name:
                uniform_target = torch.ones_like(weights) / weights.size(-1)
                kl_div = F.kl_div(
                    F.log_softmax(weights, dim=-1),
                    uniform_target,
                    reduction='batchmean'
                )
                reg_loss += kl_div
        return reg_loss


class MSALoRARegularizationLoss(nn.Module):
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config

    def forward(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        losses = {}
        total_loss = 0.0
        lora_A_params = []
        lora_B_params = []
        for name, param in model.named_parameters():
            if 'lora_A' in name and param.requires_grad:
                lora_A_params.append(param)
            elif 'lora_B' in name and param.requires_grad:
                lora_B_params.append(param)
        if lora_A_params or lora_B_params:
            l2_loss = 0.0
            for param in lora_A_params + lora_B_params:
                l2_loss += torch.norm(param, p=2)
            l2_loss *= self.config.l2_weight
            losses['l2_loss'] = l2_loss
            total_loss += l2_loss
        if lora_A_params and lora_B_params:
            ortho_loss = 0.0
            for A_param, _ in zip(lora_A_params, lora_B_params):
                if A_param.size(0) > 1:
                    A_gram = torch.matmul(A_param, A_param.t())
                    A_eye = torch.eye(A_param.size(0), device=A_param.device)
                    ortho_loss += torch.norm(A_gram - A_eye, p='fro')
            ortho_loss *= self.config.orthogonal_weight
            losses['orthogonal_loss'] = ortho_loss
            total_loss += ortho_loss
        if lora_A_params:
            sparsity_loss = 0.0
            for param in lora_A_params:
                sparsity_loss += torch.norm(param, p=1)
            sparsity_loss *= self.config.sparsity_weight
            losses['sparsity_loss'] = sparsity_loss
            total_loss += sparsity_loss
        losses['total_regularization_loss'] = total_loss
        return losses


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, ignore_index: int = -100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class MultiStageLoss(nn.Module):
    def __init__(self, config: LossConfig = None):
        super().__init__()
        if config is None:
            config = LossConfig()
        self.config = config
        self.generation_loss = GenerationLoss(config)
        self.contrastive_loss = ContrastiveLoss(config)
        self.clinical_loss = ClinicalValidationLoss(config)
        self.fusion_loss = FusionLoss(config)
        self.regularization_loss = MSALoRARegularizationLoss(config)

    def compute_loss(self, model_outputs: Dict[str, torch.Tensor], batch: Dict[str, Any], stage: int,
                     custom_weights: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        if custom_weights:
            weights = custom_weights
        else:
            weights = self.config.stage_weights.get(stage, self.config.stage_weights[3])
        losses = {}
        total_loss = 0.0
        if 'logits' in model_outputs and weights.get('generation', 0) > 0:
            gen_loss_dict = self.generation_loss(
                model_outputs['logits'],
                batch['labels'],
                batch.get('attention_mask')
            )
            losses.update({f'generation_{k}': v for k, v in gen_loss_dict.items()})
            total_loss += gen_loss_dict['loss'] * weights['generation']
        if (weights.get('contrastive', 0) > 0 and
                'vision_features' in model_outputs and 'fused_features' in model_outputs):
            vision_feats = model_outputs['vision_features']
            if len(vision_feats.shape) > 2:
                vision_feats = vision_feats.mean(dim=1)
            text_feats = model_outputs['fused_features']
            if len(text_feats.shape) > 2:
                text_feats = text_feats.mean(dim=1)
            cont_loss_dict = self.contrastive_loss(vision_feats, text_feats)
            losses.update({f'contrastive_{k}': v for k, v in cont_loss_dict.items()})
            total_loss += cont_loss_dict['loss'] * weights['contrastive']
        if weights.get('clinical', 0) > 0 and 'fused_features' in model_outputs:
            clinical_loss_dict = self.clinical_loss(
                model_outputs['fused_features'],
                batch.get('clinical_labels'),
                batch.get('generated_texts'),
                batch.get('texts')
            )
            losses.update({f'clinical_{k}': v for k, v in clinical_loss_dict.items()})
            total_loss += clinical_loss_dict['loss'] * weights['clinical']
        if weights.get('fusion', 0) > 0 and 'fusion_weights' in model_outputs:
            fusion_loss_dict = self.fusion_loss(
                model_outputs,
                model_outputs.get('fusion_weights')
            )
            losses.update({f'fusion_{k}': v for k, v in fusion_loss_dict.items()})
            if 'total_fusion_loss' in fusion_loss_dict:
                total_loss += fusion_loss_dict['total_fusion_loss'] * weights['fusion']
        if weights.get('msa_lora', 0) > 0:
            pass
        losses['total_loss'] = total_loss
        losses['stage'] = torch.tensor(stage)
        return losses

    def compute_regularization_loss(self, model: nn.Module, stage: int) -> Dict[str, torch.Tensor]:
        weights = self.config.stage_weights.get(stage, self.config.stage_weights[3])
        if weights.get('msa_lora', 0) > 0:
            reg_loss_dict = self.regularization_loss(model)
            if 'total_regularization_loss' in reg_loss_dict:
                reg_loss_dict['total_regularization_loss'] *= weights['msa_lora']
            return reg_loss_dict
        return {'total_regularization_loss': torch.tensor(0.0)}

    def get_loss_weights(self, stage: int) -> Dict[str, float]:
        return self.config.stage_weights.get(stage, self.config.stage_weights[3])

    def update_loss_weights(self, stage: int, new_weights: Dict[str, float]):
        self.config.stage_weights[stage] = new_weights


def create_loss_config(stage_weights: Optional[Dict[int, Dict[str, float]]] = None, temperature: float = 0.07,
                       clinical_weight: float = 0.5, **kwargs) -> LossConfig:
    return LossConfig(
        stage_weights=stage_weights,
        temperature=temperature,
        clinical_weight=clinical_weight,
        **kwargs
    )


def create_multi_stage_loss(use_focal_loss: bool = False, label_smoothing: float = 0.1, temperature: float = 0.07,
                            **kwargs) -> MultiStageLoss:
    config = create_loss_config(
        use_focal_loss=use_focal_loss,
        label_smoothing=label_smoothing,
        temperature=temperature,
        **kwargs
    )
    return MultiStageLoss(config)


if __name__ == "__main__":
    print("Testing Multi-Stage Loss Functions...")
    loss_config = LossConfig()
    multi_stage_loss = MultiStageLoss(loss_config)
    batch_size = 4
    seq_len = 128
    vocab_size = 32000
    feature_dim = 1024
    model_outputs = {
        'logits': torch.randn(batch_size, seq_len, vocab_size),
        'vision_features': torch.randn(batch_size, feature_dim),
        'fused_features': torch.randn(batch_size, seq_len, feature_dim),
        'fusion_weights': {
            'level_0_attention': torch.randn(batch_size, seq_len, seq_len),
            'level_1_attention': torch.randn(batch_size, seq_len, seq_len)
        }
    }
    batch = {
        'labels': torch.randint(0, vocab_size, (batch_size, seq_len)),
        'attention_mask': torch.ones(batch_size, seq_len),
        'texts': ['Sample medical report'] * batch_size
    }
    for stage in [1, 2, 3]:
        print(f"\n=== Testing Stage {stage} ===")
        losses = multi_stage_loss.compute_loss(
            model_outputs, batch, stage
        )
        print(f"Total loss: {losses['total_loss'].item():.4f}")
        print(f"Loss components:")
        for key, value in losses.items():
            if isinstance(value, torch.Tensor) and value.dim() == 0:
                print(f"  {key}: {value.item():.4f}")
        weights = multi_stage_loss.get_loss_weights(stage)
        print(f"Stage {stage} weights: {weights}")
    print("\nLoss function testing completed successfully!")
