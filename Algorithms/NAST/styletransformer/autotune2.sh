#!/usr/bin/env bash

time=$(date "+%Y-%m-%d %H:%M:%S")
log="./eval.log"


python run.py --dataid ../data/amazon --clsrestore clsshakes_best --pretrain_batch 3000 \
              --self_factor ${1} --cyc_factor ${2} --fw_adv_factor 2.5 --bw_adv_factor 2.0 --epoch 300 \
              --dataset amazon --max_length 32 --lr_F 0.0005
if [ $? -eq 0 ]
then
  echo "${time} --self_factor ${1} --cyc_factor ${2} --fw_adv_factor 1.5 --bw_adv_factor --dataset amazon 1.0 done" >> ${log}
else
  echo "${time} --self_factor ${1} --cyc_factor ${2} --fw_adv_factor 1.5 --bw_adv_factor --dataset amazon 1.0 failed" >> ${log}
fi

