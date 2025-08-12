# Process PadChest only
`
python scripts/prepare_data.py \
    --dataset padchest \
    --data_root /path/to/padchest \
    --output_dir ./processed_data
`
# Process ROCO v2 only  
`
python scripts/prepare_data.py \
    --dataset roco_v2 \
    --data_root /path/to/roco \
    --output_dir ./processed_data
`
# Process both datasets for multi-stage training
`
python scripts/prepare_data.py \
    --dataset both \
    --padchest_root /path/to/padchest \
    --roco_root /path/to/roco \
    --output_dir ./processed_data \
    --validate_images
`