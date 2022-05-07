TASK=c3
LR=3e-5
BATCH_SIZE=16
MODEL_TYPE=roformerv2
python run_c3.py \
  --model_type $MODEL_TYPE \
  --data_dir /root/autodl-tmp/GAU-PyTorch/CLUE/mrc_data/$TASK \
  --model_name_or_path /root/autodl-tmp/models/roformerv2_chinese_base \
  --task_name $TASK \
  --output_dir /root/autodl-tmp/GAU-PyTorch/CLUE/output/$TASK/$MODEL_TYPE/ \
  --max_seq_length 512 \
  --train_batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --do_train \
  --do_eval \
  --do_lower_case \
  --warmup_proportion 0.1 \
  --num_train_epochs 8 \
  --gradient_accumulation_steps 2