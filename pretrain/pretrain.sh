TRAIN_DIR=/root/autodl-tmp/GAU-PyTorch/clue_small_wwm_data
BATCH_SIZE=32
ACCUMULATION=8
LR=2e-4
ACTIVATION_FUNCTION=roformerv2
python run_mlm_wwm.py \
  --do_train \
  --model_type roformerv2 \
  --tokenizer_name /root/autodl-tmp/models/roformerv2_chinese_base \
  --train_dir $TRAIN_DIR \
  --output_dir /root/autodl-tmp/GAU-PyTorch/outputs/$ACTIVATION_FUNCTION/ \
  --logging_dir /root/tf-logs/$ACTIVATION_FUNCTION/ \
  --per_device_train_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $ACCUMULATION \
  --learning_rate $LR \
  --weight_decay 0.01 \
  --adam_epsilon 1e-6 \
  --max_steps 30000 \
  --warmup_steps 3000 \
  --logging_steps 50 \
  --save_steps 2000 \
  --seed 1234 \
  --max_grad_norm 1.0 \
  --dataloader_num_workers 6 \
  --fp16 \
  --activation_function $ACTIVATION_FUNCTION \
  --scaling_factor n \
  --pad_to_max_length False \

