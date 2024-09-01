#!/usr/bin/env bash

time=$(date "+%Y-%m-%d %H:%M:%S")
log="running.log"

decay=(0.8 0.9 0.98)
style=(1.0 1.2)

for d in "${decay[@]}" #decay_rate
do
  for s in "${style[@]}"
  do
    python main_gyafc_for2infor.py --decay ${d} --cyclic 1.0 --style ${s} --pseudo_method "lexical" --dataset "GYAFC_EM"
    if [ $? -eq 0 ]
    then
      echo "${time} decay:${d}_cyclic:1.0_style:${s} done" >> ${log}
    else
      echo "${time} decay:${d}_cyclic:1.0_style:${s} failed" >> ${log}
    fi
  done
done





