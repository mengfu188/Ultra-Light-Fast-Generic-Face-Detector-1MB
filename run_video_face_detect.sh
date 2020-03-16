#!/usr/bin/env bash
video=/home/cmf/datasets/tayg/tayg_duoren_part1.mp4
video=rtsp://admin:123456@192.168.1.20:554/profile1

model_path='models/train-helmet-version-RFB-320/RFB-Epoch-50-Loss-2.43834184328715.pth'
#model_path='models/train-helmet-version-RFB-640/RFB-Epoch-199-Loss-2.171001459757487.pth'

python run_video_face_detect.py \
--video_path ${video} \
--input_size 640 --threshold 0.8 \
--model_path ${model_path}

