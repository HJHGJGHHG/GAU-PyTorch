TRAIN_DIR=/root/autodl-tmp/GAU-PyTorch/clue_small_wwm_data
OUTPUT_DIR=/root/autodl-tmp/GAU-PyTorch/outputs
BATCH_SIZE=64
ACCUMULATION=4
LR=2e-4
python run_mlm_wwm.py \
  --do_train \
  --tokenizer_name junnyu/roformer_chinese_char_base \
  --train_dir $TRAIN_DIR \
  --output_dir $OUTPUT_DIR \
  --logging_dir /root/tf-logs/$BATCH_SIZE \
  --per_device_train_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $ACCUMULATION \
  --learning_rate $LR \
  --weight_decay 0.01 \
  --adam_epsilon 1e-6 \
  --max_steps 30000 \
  --warmup_steps 3000 \
  --logging_steps 50 \
  --save_steps 3000 \
  --seed 1234 \
  --max_grad_norm 1.0 \
  --dataloader_num_workers 6 \
  --fp16
