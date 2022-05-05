TASK=cmnli
LR=1e-5
BATCH_SIZE=8
python evaluate.py \
  --model_type gau \
  --data_dir /root/autodl-tmp/GAU-PyTorch/CLUE/CLUEdatasets/$TASK \
  --model_name_or_path /root/autodl-tmp/GAU-PyTorch/outputs/softmax/none \
  --task_name $TASK \
  --output_dir /root/autodl-tmp/GAU-PyTorch/CLUE/output/$TASK/ \
  --max_seq_length 512 \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --do_train \
  --do_eval \
  --do_lower_case \
  --overwrite_output_dir \
  --logging_steps 20 \
  --warmup_proportion 0.1 \
  --num_train_epochs 30
