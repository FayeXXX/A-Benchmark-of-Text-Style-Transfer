#!/usr/bin/env bash

log="./eval.log"
time=$(date "+%Y-%m-%d %H:%M:%S")

for fw in {1.5,1.0}
do
  python run.py --dataid ../data/yelp --clsrestore cls_yelp_best --use_learnable --pretrain_batch 1000 \
                --self_factor ${1} --cyc_factor ${2} --fw_adv_factor ${fw} --bw_adv_factor 1.5 --epoch 300 \
                --dataset yelp
  if [ $? -eq 0 ]
  then
    echo "${time} --self_factor ${1} --cyc_factor ${2} --fw_adv_factor ${fw} --bw_adv_factor 1.5 --dataset yelp done" >> ${log}
  else
    echo "${time} --self_factor ${1} --cyc_factor ${2} --fw_adv_factor ${fw} --bw_adv_factor 1.5 --dataset yelp failed" >> ${log}
  fi
done
