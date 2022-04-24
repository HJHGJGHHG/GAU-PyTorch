TASK=cmnli
LR=3e-5
BATCH_SIZE=32
python evaluate.py \
  --model_type gau \
  --data_dir /root/autodl-tmp/GAU-PyTorch/CLUE/CLUEdatasets/$TASK \
  --model_name_or_path /root/autodl-tmp/models/GAU-Base-Full \
  --task_name $TASK \
  --output_dir /root/autodl-tmp/GAU-PyTorch/CLUE/output/$TASK/ \
  --max_seq_length 512 \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --do_train \
  --do_eval \
  --do_lower_case \
  --overwrite_output_dir \
  --logging_steps 500 \
  --warmup_proportion 0.1 \
  --num_train_epochs 3
