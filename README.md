# Radixpert: A Staged Adaptation and Hierarchical Fusion Framework for Radiology VLMs
[![PDF](https://img.shields.io/badge/PDF-Download-red.svg)](https://assets.tina.io/1fb09d03-9237-4c49-aaa9-d024a83c7ac7/Radixpert__A_Staged_Adaptation_and_Hierarchical_Fusion_Framework_for_Radiology_VLMs.pdf)

## Abstract

Automated radiology report generation has emerged as a critical application for improving clinical efficiency and reducing physician workload. This paper presents Radixpert, a novel vision-language model that leverages multi-dataset training and hierarchical cross-modal fusion for enhanced radiology report generation. Our approach utilizes the Llama 3.2-11B-Vision-Instruct model as the foundation, enhanced with a Multi Stage Adaptive LoRA (MSA-LoRA) fine-tuning methodology and a Hierarchical Cross-Modal Fusion (HCF) architecture. We trained our model on a combination of ROCO v2 (15,000 samples) and PadChest (16,000 samples) datasets, achieving state-of-the- art performance across multiple evaluation metrics. Radixpert demonstrates superior clinical accuracy with a BLEU-4 score of 0.194, CIDEr of 0.478, and RadCliQ-v1 of 0.823, outperforming existing methods. Code and model weights will be made available upon publication to support reproducible research in medical AI.

## Quick Start
`
python scripts/train.py \
    --padchest_root /data/padchest \
    --roco_root /data/roco_v2 \
    --output_dir ./radixpert_experiments/run1
`

### Training with multiple args

`
python scripts/train.py \
    --padchest_root /data/padchest \
    --roco_root /data/roco_v2 \
    --output_dir ./radixpert_experiments/high_lr \
    --stages "1,2,3" \
    --batch_size 4 \
    --num_epochs 15 20 25 \
    --learning_rate 1e-4 5e-5 2e-5 \
    --lora_rank 16 24 32 \
    --mixed_precision \
    --use_sam \
    --wandb_project "radixpert-ablation"
`

### To Resume Training

`
python scripts/train.py \
    --padchest_root /data/padchest \
    --roco_root /data/roco_v2 \
    --output_dir ./radixpert_experiments/run1 \
    --auto_resume
`
### Installation
`pip3 install requirements.txt`