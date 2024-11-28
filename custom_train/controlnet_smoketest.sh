accelerate launch train_controlnet.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
  --output_dir=runs/ \
  --dataset_name="JFoz/dog-poses-controlnet-dataset" \
  --resolution=128 \
  --learning_rate=1e-5 \
  --train_batch_size=64 \
  --dataloader_num_workers=8 \
  --report_to=wandb \
  --checkpointing_steps=500 \
  --validation_steps=500 \
  --max_train_steps=3000 \
  --proportion_empty_prompts=0.5 \
  --gradient_checkpointing \
  --gradient_accumulation_steps=1 \
  --resume_from_checkpoint=latest \
  --allow_tf32 \
  --validation_image "./conditioning_image1.jpg" "./conditioning_image2.jpg" \
  --validation_prompt "a small yorkshire terrier dog is sitting on a black background" "a black dog with a toy in its mouth on the beach" \
  --image_column "original_image" \
  --conditioning_image_column "conditioning_image" \
  --caption_column "caption" \
  # --use_8bit_adam 
  # --add_mask \
  # --mask_weight=0.2 \
#   --aug_dataset_name="/path/to/your/jsonl/data.jsonl" \
#   --mixed_precision fp16 \