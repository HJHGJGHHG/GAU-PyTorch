LR=2e-5
BATCH_SIZE=16
MODEL_TYPE=roformerv2
python run_mrc.py \
  --model_type $MODEL_TYPE \
  --model_name_or_path /root/autodl-tmp/models/roformerv2_chinese_base\
  --train_epochs=3 \
  --n_batch=$BATCH_SIZE \
  --lr=$LR \
  --warmup_rate=0.1 \
  --max_seq_length=512 \
  --checkpoint_dir /root/autodl-tmp/GAU-PyTorch/CLUE/output/CMRC/$MODEL_TYPE \
  --eval_epochs 0.2 \


