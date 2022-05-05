TASK=c3
LR=3e-5
BATCH_SIZE=32
python run_c3.py \
  --model_type gau \
  --data_dir /root/autodl-tmp/GAU-PyTorch/CLUE/mrc_data/$TASK \
  --model_name_or_path /root/autodl-tmp/GAU-PyTorch/outputs/softmax/none \
  --task_name $TASK \
  --output_dir /root/autodl-tmp/GAU-PyTorch/CLUE/output/$TASK/ \
  --max_seq_length 512 \
  --train_batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --do_train \
  --do_eval \
  --do_lower_case \
  --warmup_proportion 0.1 \
  --num_train_epochs 6