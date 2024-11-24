accelerate launch train_controlnet.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
  --output_dir=dilightnet-openillum/ \
  --dataset_name="dataset/train_data_final.jsonl" \
  --validation_dataset_name="dataset/eval_data_final.jsonl" \
  --resolution=128 \
  --shading_hint_channels=12 \
  --learning_rate=1e-5 \
  --train_batch_size=64 \
  --dataloader_num_workers=8 \
  --report_to=wandb \
  --checkpointing_steps=500 \
  --validation_steps=1000 \
  --max_train_steps=3000 \
  --proportion_empty_prompts=0.1 \
  --proportion_channel_aug=0.2 \
  --gradient_checkpointing \
  --gradient_accumulation_steps=1 \
  --set_grads_to_none \
  --resume_from_checkpoint=latest \
  --allow_tf32 \
  --num_validation_images=8 \
  --add_mask \
  --checkpoints_total_limit=4 \
  --tracker_project_name="dilightnet-openillum" \
  # --use_8bit_adam 
  # --mask_weight=0.2 \
#   --aug_dataset_name="/path/to/your/jsonl/data.jsonl" \
#   --mixed_precision fp16 \