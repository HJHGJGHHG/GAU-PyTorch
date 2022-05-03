TRAIN_DIR=/root/autodl-tmp/GAU-PyTorch/small_data
BATCH_SIZE=128
ACCUMULATION=8
LR=1e-3
ACTIVATION_FUNCTION=squared_relu
SCALING_FACTOR=ns
python run_mlm_wwm.py \
  --do_train \
  --tokenizer_name /root/autodl-tmp/models/GAU-Base-Full \
  --train_dir $TRAIN_DIR \
  --output_dir /root/autodl-tmp/GAU-PyTorch/outputs/$ACTIVATION_FUNCTION/$SCALING_FACTOR \
  --logging_dir /root/tf-logs/$ACTIVATION_FUNCTION/$SCALING_FACTOR \
  --per_device_train_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $ACCUMULATION \
  --learning_rate $LR \
  --weight_decay 0.01 \
  --adam_epsilon 1e-6 \
  --max_steps 10000 \
  --warmup_steps 1000 \
  --logging_steps 10 \
  --save_steps 1000 \
  --seed 1234 \
  --max_grad_norm 1.0 \
  --dataloader_num_workers 6 \
  --fp16 \
  --activation_function $ACTIVATION_FUNCTION \
  --scaling_factor $SCALING_FACTOR
