TRAIN_DIR=/root/autodl-tmp/GAU-PyTorch/clue_small_wwm_data
OUTPUT_DIR=/root/autodl-tmp/GAU-PyTorch/squared_relu/
BATCH_SIZE=128
ACCUMULATION=4
LR=3e-4
python run_mlm_wwm.py \
  --do_train \
  --tokenizer_name junnyu/roformer_chinese_char_base \
  --train_dir $TRAIN_DIR \
  --output_dir $OUTPUT_DIR \
  --logging_dir /root/tf-logs \
  --per_device_train_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $ACCUMULATION \
  --learning_rate $LR \
  --weight_decay 0.01 \
  --adam_epsilon 1e-6 \
  --max_steps 50000 \
  --warmup_steps 5000 \
  --logging_steps 100 \
  --save_steps 5000 \
  --seed 1234 \
  --max_grad_norm 3.0 \
  --dataloader_num_workers 6 \
  --fp16 \
  --overwrite_output_dir
