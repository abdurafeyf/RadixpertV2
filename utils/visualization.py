import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import cv2
from PIL import Image
import logging


class RadixpertVisualizer:
    def __init__(self, output_dir: Union[str, Path] = "./visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use('default')
        sns.set_palette("husl")
        
        self.logger = logging.getLogger("RadixpertVisualizer")
    
    def visualize_attention_maps(
        self,
        image: torch.Tensor,
        attention_weights: Dict[str, torch.Tensor],
        text_tokens: List[str],
        save_path: Optional[str] = None
    ):
        num_layers = len(attention_weights)
        fig, axes = plt.subplots(2, num_layers, figsize=(4*num_layers, 8))
        
        if num_layers == 1:
            axes = axes.reshape(2, 1)
        
        if image.dim() == 3:
            image_np = image.permute(1, 2, 0).cpu().numpy()
        else:
            image_np = image.cpu().numpy()
        
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        
        for i, (layer_name, attn) in enumerate(attention_weights.items()):
            if attn.dim() > 2:
                attn = attn.mean(dim=0)
                if attn.size(0) == attn.size(1):
                    attn = attn.diag()
            
            attn_map = self._reshape_attention_to_image(attn, image_np.shape[:2])
            
            axes[0, i].imshow(image_np, cmap='gray')
            axes[0, i].set_title(f'{layer_name} - Original')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(image_np, cmap='gray')
            im = axes[1, i].imshow(attn_map, cmap='jet', alpha=0.6)
            axes[1, i].set_title(f'{layer_name} - Attention')
            axes[1, i].axis('off')
            
            plt.colorbar(im, ax=axes[1, i], fraction=0.046)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Attention visualization saved: {save_path}")
        
        return fig
    
    def plot_training_progress(
        self,
        training_history: Dict[str, List[float]],
        save_path: Optional[str] = None
    ):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        if 'train_loss' in training_history and 'val_loss' in training_history:
            axes[0, 0].plot(training_history['train_loss'], label='Train Loss')
            axes[0, 0].plot(training_history['val_loss'], label='Validation Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        if 'bleu_4' in training_history:
            axes[0, 1].plot(training_history['bleu_4'], label='BLEU-4', color='green')
            axes[0, 1].set_title('BLEU-4 Score Progress')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('BLEU-4')
            axes[0, 1].grid(True)
        
        clinical_metrics = ['radcliq_v1', 'radgraph_f1', 'clinical_accuracy']
        colors = ['red', 'blue', 'orange']
        for metric, color in zip(clinical_metrics, colors):
            if metric in training_history:
                axes[1, 0].plot(training_history[metric], label=metric.replace('_', ' ').title(), color=color)
        
        axes[1, 0].set_title('Clinical Metrics Progress')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        if 'learning_rate' in training_history:
            axes[1, 1].plot(training_history['learning_rate'], label='Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training progress plot saved: {save_path}")
        
        return fig
    
    def visualize_feature_maps(
        self,
        feature_maps: Dict[str, torch.Tensor],
        max_channels: int = 16,
        save_path: Optional[str] = None
    ):
        num_layers = len(feature_maps)
        fig, axes = plt.subplots(num_layers, max_channels, figsize=(2*max_channels, 2*num_layers))
        
        if num_layers == 1:
            axes = axes.reshape(1, -1)
        
        for layer_idx, (layer_name, feat) in enumerate(feature_maps.items()):
            feat_np = feat.detach().cpu().numpy()
            if feat_np.ndim == 4:
                feat_np = feat_np[0]
            if feat_np.ndim == 3:
                num_channels = min(feat_np.shape[0], max_channels)
                for ch in range(num_channels):
                    channel_data = feat_np[ch]
                    channel_data = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min() + 1e-8)
                    axes[layer_idx, ch].imshow(channel_data)
                    axes[layer_idx, ch].set_title(f'{layer_name}\nCh {ch}')
                    axes[layer_idx, ch].axis('off')
                for ch in range(num_channels, max_channels):
                    axes[layer_idx, ch].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Feature maps visualization saved: {save_path}")
        
        return fig
    
    def analyze_generated_reports(
        self,
        predictions: List[str],
        references: List[str],
        metrics: Dict[str, float],
        save_path: Optional[str] = None
    ):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        pred_lens = [len(p.split()) for p in predictions]
        ref_lens = [len(r.split()) for r in references]
        
        axes[0, 0].hist(pred_lens, bins=20, alpha=0.7, label='Generated')
        axes[0, 0].hist(ref_lens, bins=20, alpha=0.7, label='Reference')
        axes[0, 0].set_title('Report Length Distribution')
        axes[0, 0].set_xlabel('Number of Words')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        metric_names = list(metrics.keys())[:8]
        metric_values = [metrics[name] for name in metric_names]
        axes[0, 1].bar(range(len(metric_names)), metric_values)
        axes[0, 1].set_title('Evaluation Metrics')
        axes[0, 1].set_xlabel('Metric')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_xticks(range(len(metric_names)))
        axes[0, 1].set_xticklabels(metric_names, rotation=45, ha='right')
        
        from collections import Counter
        pred_words = ' '.join(predictions).lower().split()
        ref_words = ' '.join(references).lower().split()
        pred_freq = Counter(pred_words)
        ref_freq = Counter(ref_words)
        top_pred = pred_freq.most_common(10)
        top_ref = ref_freq.most_common(10)
        
        axes[1, 0].bar(range(len(top_pred)), [c for _, c in top_pred])
        axes[1, 0].set_title('Top Words in Generated Reports')
        axes[1, 0].set_xticks(range(len(top_pred)))
        axes[1, 0].set_xticklabels([w for w, _ in top_pred], rotation=45, ha='right')
        axes[1, 0].set_ylabel('Frequency')
        
        axes[1, 1].bar(range(len(top_ref)), [c for _, c in top_ref])
        axes[1, 1].set_title('Top Words in Reference Reports')
        axes[1, 1].set_xticks(range(len(top_ref)))
        axes[1, 1].set_xticklabels([w for w, _ in top_ref], rotation=45, ha='right')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Report analysis saved: {save_path}")
        
        return fig
    
    def _reshape_attention_to_image(
        self,
        attention: torch.Tensor,
        target_shape: Tuple[int, int]
    ):
        attn_np = attention.detach().cpu().numpy()
        if attn_np.ndim == 1:
            size = int(np.sqrt(len(attn_np)))
            if size * size == len(attn_np):
                attn_np = attn_np.reshape(size, size)
            else:
                attn_np = np.ones(target_shape) * attn_np.mean()
        if attn_np.shape != target_shape:
            attn_np = cv2.resize(attn_np, (target_shape[1], target_shape[0]))
        return attn_np


def create_visualizer(output_dir: str = "./visualizations"):
    return RadixpertVisualizer(output_dir)
