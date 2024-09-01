#!/usr/bin/env bash

log="./eval.log"
python run.py --name gyafc_em --dataid ../data/GYAFC_EM --clsrestore clsem_best --dataset gyfac_em --pretrain_batch 3000 \
              --self_factor ${1} --cyc_factor ${2} --fw_adv_factor 1.5 --bw_adv_factor 1.5 --epoch 400 \
              --max_length 32 --lr_F 0.0005

if [ $? -eq 0 ]
then
  echo "${time} --self_factor ${1} --cyc_factor ${2} --fw_adv_factor 1.5 --bw_adv_factor 1.5 --dataset gyafc_em done" >> ${log}
else
  echo "${time} --self_factor ${1} --cyc_factor ${2} --fw_adv_factor 1.5 --bw_adv_factor 1.5 --dataset gyafc_em failed" >> ${log}
fi

