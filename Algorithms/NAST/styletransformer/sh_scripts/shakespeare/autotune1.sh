#!/usr/bin/env bash

time=$(date "+%Y-%m-%d %H:%M:%S")
log="./eval.log"


python run.py --dataid ../data/shakespeare --clsrestore clsshakes_best --pretrain_batch 1000 \
              --self_factor ${1} --cyc_factor ${2} --fw_adv_factor 1.0 --bw_adv_factor 1.0 --epoch 300 \
              --dataset shakespeare --max_length 32 --lr_F 0.0001

if [ $? -eq 0 ]
then
  echo "${time} --self_factor ${1} --cyc_factor ${2} --fw_adv_factor 1.0 --bw_adv_factor 1.0 --dataset shakes done" >> ${log}
else
  echo "${time} --self_factor ${1} --cyc_factor ${2} --fw_adv_factor 1.0 --bw_adv_factor 1.0 --dataset shakes failed" >> ${log}
fi

