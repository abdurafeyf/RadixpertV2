import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaTokenizer
from typing import Dict, List, Optional, Tuple, Union
import math


from msa_lora import MSALoRAConfig, MSALoRALayer
from hcf_fusion import HierarchicalCrossModalFusion
from vision_encoder import VisionEncoder


class RadixpertConfig:
    
    def __init__(
        self,
        llama_model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
        max_length: int = 2048,
        
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: List[str] = None,
        
        fusion_layers: int = 3,
        fusion_dim: int = 4096,
        vision_dim: int = 1024,
        text_dim: int = 4096,
        
        stage: int = 1,
        freeze_vision_encoder: bool = False,
        freeze_text_decoder: bool = False,
        
        max_report_length: int = 512,
        enable_clinical_validation: bool = True,
    ):
        self.llama_model_name = llama_model_name
        self.max_length = max_length
        
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        
        self.fusion_layers = fusion_layers
        self.fusion_dim = fusion_dim
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        
        self.stage = stage
        self.freeze_vision_encoder = freeze_vision_encoder
        self.freeze_text_decoder = freeze_text_decoder
        
        self.max_report_length = max_report_length
        self.enable_clinical_validation = enable_clinical_validation



class Radixpert(nn.Module):
    
    def __init__(self, config: RadixpertConfig):
        super().__init__()
        self.config = config
        
        self.llama_model = LlamaForCausalLM.from_pretrained(
            config.llama_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.tokenizer = LlamaTokenizer.from_pretrained(config.llama_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.vision_encoder = VisionEncoder(
            vision_dim=config.vision_dim,
            output_dim=config.fusion_dim
        )
        
        self._init_msa_lora()
        
        self.hcf_fusion = HierarchicalCrossModalFusion(
            vision_dim=config.vision_dim,
            text_dim=config.text_dim,
            fusion_dim=config.fusion_dim,
            num_layers=config.fusion_layers
        )
        
        self.report_head = nn.Sequential(
            nn.Linear(config.text_dim, config.text_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.text_dim // 2, self.llama_model.config.vocab_size)
        )
        
        self._configure_training_stage()
        
        self._add_medical_tokens()
    
    def _init_msa_lora(self):
        msa_config = MSALoRAConfig(
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            stage=self.config.stage
        )
        
        for name, module in self.llama_model.named_modules():
            if any(target in name for target in self.config.target_modules):
                if isinstance(module, nn.Linear):
                    lora_layer = MSALoRALayer(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        config=msa_config,
                        original_layer=module
                    )
                    self._set_module_by_name(name, lora_layer)
    
    def _set_module_by_name(self, name: str, module: nn.Module):
        names = name.split('.')
        parent = self.llama_model
        for name_part in names[:-1]:
            parent = getattr(parent, name_part)
        setattr(parent, names[-1], module)
    
    def _configure_training_stage(self):
        if self.config.stage == 1:
            for param in self.llama_model.parameters():
                param.requires_grad = False
            
        elif self.config.stage == 2:
            for param in self.llama_model.parameters():
                param.requires_grad = False
            
        elif self.config.stage == 3:
            for param in self.llama_model.parameters():
                param.requires_grad = True
        
        for param in self.hcf_fusion.parameters():
            param.requires_grad = True
        for param in self.vision_encoder.parameters():
            param.requires_grad = True
    
    def _add_medical_tokens(self):
        medical_tokens = [
            "<FINDING>", "</FINDING>",
            "<IMPRESSION>", "</IMPRESSION>",
            "<TECHNIQUE>", "</TECHNIQUE>",
            "<COMPARISON>", "</COMPARISON>",
            "<INDICATION>", "</INDICATION>",
            "<ANATOMY>", "</ANATOMY>",
            "<PATHOLOGY>", "</PATHOLOGY>",
            "<NORMAL>", "<ABNORMAL>",
            "<URGENT>", "<ROUTINE>"
        ]
        
        new_tokens = [token for token in medical_tokens if token not in self.tokenizer.vocab]
        if new_tokens:
            self.tokenizer.add_tokens(new_tokens)
            self.llama_model.resize_token_embeddings(len(self.tokenizer))
    
    def forward(
        self,
        images: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_fusion_weights: bool = False
    ) -> Dict[str, torch.Tensor]:
        batch_size = images.size(0)
        
        vision_features = self.vision_encoder(images)
        
        if input_ids is not None:
            text_embeddings = self.llama_model.model.embed_tokens(input_ids)
        else:
            text_embeddings = None
        
        if text_embeddings is not None:
            fused_features, fusion_weights = self.hcf_fusion(
                vision_features, text_embeddings, attention_mask
            )
        else:
            fused_features = self.hcf_fusion.vision_only_forward(vision_features)
            fusion_weights = None
        
        if input_ids is not None:
            outputs = self.llama_model(
                inputs_embeds=fused_features,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
            
            result = {
                'loss': outputs.loss,
                'logits': outputs.logits,
                'vision_features': vision_features,
                'fused_features': fused_features
            }
        else:
            outputs = self.llama_model.generate(
                inputs_embeds=fused_features,
                max_length=self.config.max_report_length,
                num_beams=4,
                early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True
            )
            
            result = {
                'generated_ids': outputs.sequences,
                'vision_features': vision_features,
                'fused_features': fused_features
            }
        
        if return_fusion_weights and fusion_weights is not None:
            result['fusion_weights'] = fusion_weights
            
        return result
    
    def generate_report(
        self,
        images: torch.Tensor,
        prompt: Optional[str] = None,
        max_length: int = 512,
        temperature: float = 0.7,
        num_beams: int = 4
    ) -> List[str]:
        self.eval()
        
        with torch.no_grad():
            if prompt is not None:
                prompt_ids = self.tokenizer.encode(
                    prompt, 
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(images.device)
                
                attention_mask = (prompt_ids != self.tokenizer.pad_token_id).long()
                
                outputs = self.forward(
                    images=images,
                    input_ids=prompt_ids,
                    attention_mask=attention_mask
                )
                
                generated_ids = self.llama_model.generate(
                    inputs_embeds=outputs['fused_features'],
                    max_length=max_length,
                    temperature=temperature,
                    num_beams=num_beams,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=temperature > 0
                )
            else:
                outputs = self.forward(images=images)
                generated_ids = outputs['generated_ids']
            
            reports = []
            for ids in generated_ids:
                report = self.tokenizer.decode(ids, skip_special_tokens=True)
                reports.append(report)
            
            return reports
    
    def get_trainable_parameters(self) -> Dict[str, int]:
        counts = {
            'total': 0,
            'lora': 0,
            'fusion': 0,
            'vision_encoder': 0,
            'report_head': 0
        }
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                param_count = param.numel()
                counts['total'] += param_count
                
                if 'lora' in name.lower():
                    counts['lora'] += param_count
                elif 'hcf_fusion' in name:
                    counts['fusion'] += param_count
                elif 'vision_encoder' in name:
                    counts['vision_encoder'] += param_count
                elif 'report_head' in name:
                    counts['report_head'] += param_count
        
        return counts
    
    def save_stage_checkpoint(self, path: str, stage: int):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'stage': stage,
            'trainable_params': self.get_trainable_parameters()
        }
        torch.save(checkpoint, path)
    
    def load_stage_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint['stage']



def create_radixpert_model(
    stage: int = 1,
    lora_rank: int = 16,
    fusion_layers: int = 3,
    **kwargs
) -> Radixpert:
    config = RadixpertConfig(
        stage=stage,
        lora_rank=lora_rank,
        fusion_layers=fusion_layers,
        **kwargs
    )
    
    model = Radixpert(config)
    
    print(f"Created Radixpert model for Stage {stage}")
    print(f"Trainable parameters: {model.get_trainable_parameters()}")
    
    return model



if __name__ == "__main__":
    print("Creating Radixpert models for different training stages...")
    
    model_stage1 = create_radixpert_model(stage=1, lora_rank=16)
    
    model_stage2 = create_radixpert_model(stage=2, lora_rank=16)
    
    model_stage3 = create_radixpert_model(stage=3, lora_rank=16)
    
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        reports = model_stage3.generate_report(
            images, 
            prompt="FINDINGS: ",
            max_length=256
        )
        
        for i, report in enumerate(reports):
            print(f"Generated Report {i+1}: {report}")
