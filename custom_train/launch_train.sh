accelerate launch DiLightNet/train/train_controlnet.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
  --output_dir=runs/ \
  --exp_id="pre-exp3" \
  --dataset_name="dataset/train_data_final.jsonl" \
  --test_dataset_name="dataset/eval_data_final.jsonl" \
  --resolution=128 \
  --shading_hint_channels=12 \
  --learning_rate=1e-5 \
  --train_batch_size=64 \
  --dataloader_num_workers=8 \
  --report_to=wandb \
  --checkpointing_steps=500 \
  --validation_steps=1000 \
  --max_train_steps=3000 \
  --proportion_empty_prompts=0.5 \
  --proportion_channel_aug=0.2 \
  --proportion_pred_normal=0.1 \
  --gradient_checkpointing \
  --gradient_accumulation_steps=1 \
  --set_grads_to_none \
  --resume_from_checkpoint=latest \
  --allow_tf32
  # --use_8bit_adam 
  # --add_mask \
  # --mask_weight=0.2 \
#   --aug_dataset_name="/path/to/your/jsonl/data.jsonl" \
#   --mixed_precision fp16 \