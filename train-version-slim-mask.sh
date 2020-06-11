#!/usr/bin/env bash
model_root_path="/checkpoints/face_detect/train-version-slim-640"
log_dir="$model_root_path/logs"
log="$log_dir/log"
mkdir -p "$log_dir"

python3 -u train.py \
  --datasets \
  /datasets/widerface/wider_face_add_lm_10_10 \
  --validation_dataset \
  /datasets/widerface/wider_face_add_lm_10_10 \
  --net \
  slim \
  --num_epochs \
  200 \
  --milestones \
  95 150 \
  --lr \
  1e-2 \
  --batch_size \
  24 \
  --input_size \
  640 \
  --checkpoint_folder \
  ${model_root_path} \
  --num_workers \
  24 \
  --log_dir \
  ${log_dir} \
  --cuda_index \
  "0,1,2,3" \
  2>&1 | tee "$log"
