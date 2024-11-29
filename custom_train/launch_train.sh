accelerate launch train_controlnet.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
  --output_dir=runs/dilightnet-openillum-2-1-2-base-v2 \
  --dataset_name="dataset_v2/train_v2.jsonl" \
  --validation_dataset_name="dataset_v2/eval_v2.jsonl" \
  --resolution=128 \
  --shading_hint_channels=12 \
  --learning_rate=1e-5 \
  --lr_scheduler="linear" \
  --train_batch_size=256 \
  --dataloader_num_workers=8 \
  --report_to=wandb \
  --checkpointing_steps=1000 \
  --validation_steps=250 \
  --max_train_steps=10000 \
  --proportion_empty_prompts=0.1 \
  --proportion_channel_aug=0 \
  --gradient_checkpointing \
  --gradient_accumulation_steps=1 \
  --set_grads_to_none \
  --allow_tf32 \
  --num_validation_images=4 \
  --add_mask \
  --checkpoints_total_limit=5 \
  --tracker_project_name="dilightnet-openillum-main-exp" \
  --controlnet_model_name_or_path="runs/dilightnet-openillum-2-1-2-base-v2/checkpoint-10000/controlnet"
  # --resume_from_checkpoint=latest \
  # --resume_from_checkpoint="latest" \

# --use_8bit_adam 
# --mask_weight=0.2 \
#   --aug_dataset_name="/path/to/your/jsonl/data.jsonl" \
#   --mixed_precision fp16 \