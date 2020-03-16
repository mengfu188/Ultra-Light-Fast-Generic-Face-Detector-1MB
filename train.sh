work_dir=models/train-helmet-version-RFB-640/
mkdir ${work_dir}

python -u train.py \
--net RFB \
--dataset_type voc \
--datasets /home/cmf/share/HelmetMerge \
--validation_dataset /home/cmf/share/HelmetMerge \
--checkpoint_folder ${work_dir} \
--num_workers 6 \
--batch_size 24 \
--num_epochs 200 \
--validation_epochs 5 \
--input_size 640 \
--milestones 50 70 \
--resume models/train-helmet-version-RFB-320/RFB-Epoch-50-Loss-2.43834184328715.pth \
| tee ${work_dir}log
