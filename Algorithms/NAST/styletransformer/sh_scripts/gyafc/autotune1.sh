#!/usr/bin/env bash

time=$(date "+%Y-%m-%d %H:%M:%S")
log="./eval.log"


python run.py --dataid ../data/GYAFC_family --clsrestore cls_GYAFC_posw2_best --pretrain_batch 3000 \
              --self_factor ${1} --cyc_factor ${2} --fw_adv_factor 1.0 --bw_adv_factor 1.0 --epoch 400 \
              --dataset gyfac_fr --max_length 32 --lr_F 0.0005
if [ $? -eq 0 ]
then
  echo "${time} --self_factor ${1} --cyc_factor ${2} --fw_adv_factor 1.0 --bw_adv_factor 1.0 --dataset gyafc_fr done" >> ${log}
else
  echo "${time} --self_factor ${1} --cyc_factor ${2} --fw_adv_factor 1.0 --bw_adv_factor 1.0 --dataset gyafc_fr failed" >> ${log}
fi

