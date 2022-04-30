TASK=tnews
LR=1e-4
BATCH_SIZE=128
python task_specific.py \
  --teacher_model_type roformerv2 \
  --data_dir /root/autodl-tmp/GAU-PyTorch/CLUE/CLUEdatasets/$TASK \
  --teacher_model_name_or_path /root/autodl-tmp/models/roformerv2_chinese_base \
  --student_model_name_or_path /root/autodl-tmp/models/GAU-Base-Full/ \
  --task_name $TASK \
  --output_dir /root/autodl-tmp/GAU-PyTorch/distil/output/$TASK/ \
  --log_dir /root/tf-logs/$TASK/$BATCH_SIZE \
  --max_seq_length 512 \
  --train_batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --do_lower_case \
  --overwrite_output_dir \
  --logging_steps 1000 \
  --warmup_proportion 0.1 \
  --num_train_epochs 5  \
