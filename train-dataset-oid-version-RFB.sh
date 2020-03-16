#!/usr/bin/env bash

model_root_path="./models/train-dataset-oid-version-RFB"
log_dir="$model_root_path/logs"
log="$log_dir/log"
mkdir -p "$log_dir"

#python -u train.py \
#--datasets /home/cmf/datasets/open-image/OID \
#--dataset_type oid --net RFB --num_epochs 200 \
#--milestones 25,150 --lr 1e-2 --batch_size 24 \
#--input_size 320 --checkpoint_folder ${model_root_path} \
#--log_dir ${log_dir} --cuda_index 0 \
#--train_dataset_percentage 1 \
#2>&1 | tee "$log"

model_root_path="./models/train-dataset-oid-version-RFB_v2"
log_dir="$model_root_path/logs"
log="$log_dir/log"
mkdir -p "$log_dir"
#
#python -u train.py \
#--datasets /home/cmf/datasets/open-image/OID \
#--dataset_type oid --net RFB --num_epochs 200 \
#--milestones 25,150 --lr 1e-3 --batch_size 24 \
#--input_size 320 --checkpoint_folder ${model_root_path} \
#--log_dir ${log_dir} --cuda_index 0 \
#--train_dataset_percentage 1 \
#--num_workers 6 \
#--oid_fileter_size 10 \
#--validation_epochs 1 \
#--resume models/train-dataset-oid-version-RFB/RFB-Epoch-90-Loss-3.197492961465877.pth \
#2>&1 | tee "$log"

model_root_path="./models/train-dataset-oid-version-RFB_v3"
log_dir="$model_root_path/logs"
log="$log_dir/log"
mkdir -p "$log_dir"
#
#python -u train.py \
#--datasets /home/cmf/datasets/open-image/OID \
#--dataset_type oid --net RFB --num_epochs 200 \
#--milestones 10,150 --lr 1e-3 --batch_size 24 \
#--input_size 320 --checkpoint_folder ${model_root_path} \
#--log_dir ${log_dir} --cuda_index 0 \
#--train_dataset_percentage 1 \
#--num_workers 6 \
#--oid_fileter_size 10 \
#--validation_epochs 1 \
#--resume models/train-dataset-oid-version-RFB_v2/RFB-Epoch-7-Loss-3.00314796231959.pth \
#2>&1 | tee "$log"

# milestons 25-30

model_root_path="./models/train-dataset-oid-version-RFB-640"
log_dir="$model_root_path/logs"
log="$log_dir/log"
mkdir -p "$log_dir"

#python -u train.py \
#--datasets /home/cmf/datasets/open-image/OID \
#--dataset_type oid --net RFB --num_epochs 200 \
#--milestones 20,30 --lr 1e-2 --batch_size 24 \
#--input_size 640 --checkpoint_folder ${model_root_path} \
#--log_dir ${log_dir} --cuda_index 0 \
#--train_dataset_percentage 1 \
#--num_workers 6 \
#--oid_fileter_size 10 \
#--validation_epochs 1 \
#--resume models/train-dataset-oid-version-RFB_v3/RFB-Epoch-12-Loss-2.957992191732365.pth \
#2>&1 | tee "$log"

model_root_path="./models/train-dataset-oid-version-RFB-640-pad"
log_dir="$model_root_path/logs"
log="$log_dir/log"
mkdir -p "$log_dir"

#python -u train.py \
#--datasets /home/cmf/datasets/open-image/OID \
#--dataset_type oid --net RFB --num_epochs 200 \
#--milestones 20,30 --lr 1e-2 --batch_size 24 \
#--input_size 640 --checkpoint_folder ${model_root_path} \
#--log_dir ${log_dir} --cuda_index 0 \
#--train_dataset_percentage 1 \
#--num_workers 6 \
#--oid_fileter_size 10 \
#--validation_epochs 1 \
#--resume models/train-dataset-oid-version-RFB_v3/RFB-Epoch-12-Loss-2.957992191732365.pth \
#2>&1 | tee "$log"

model_root_path="./models/train-dataset-oid-version-RFB-320-pad-without-resume"
log_dir="$model_root_path/logs"
log="$log_dir/log"
mkdir -p "$log_dir"

python -u train.py \
--datasets /home/cmf/datasets/open-image/OID \
--validation_dataset /home/cmf/datasets/open-image/OID \
--dataset_type oid --net RFB --num_epochs 200 \
--milestones 30,50 --lr 1e-2 --batch_size 24 \
--input_size 320 --checkpoint_folder ${model_root_path} \
--log_dir ${log_dir} --cuda_index 0 \
--train_dataset_percentage 1 \
--num_workers 6 \
--oid_filter_size 0 \
--validation_epochs 1 \
2>&1 | tee "$log"

#python -u train.py \
#--datasets /home/cmf/datasets/brainwash/data_list_brainwash_train.pkl \
#--validation_dataset /home/cmf/datasets/brainwash/data_list_brainwash_test.pkl \
#--dataset_type brain --net RFB --num_epochs 200 \
#--milestones 20,30 --lr 1e-2 --batch_size 24 \
#--input_size 320 --checkpoint_folder ${model_root_path} \
#--log_dir ${log_dir} --cuda_index 0 \
#--train_dataset_percentage 0.01 \
#--num_workers 1 \
#--oid_filter_size 10 \
#--brain_filter_size 10 \
#--validation_epochs 1 \
#--resume models/train-dataset-oid-version-RFB-640-pad/RFB-Epoch-63-Loss-2.7645139572394157.pth \
#2>&1 | tee "$log"
