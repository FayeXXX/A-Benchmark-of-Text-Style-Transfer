#!/usr/bin/env bash
export LC_ALL=C.UTF-8
export LANG=C.UTF-8


alg=$1
dataset=$2
evaluate_path=./outputs/$alg/${dataset}
files=$(ls $evaluate_path)
mkdir -p "./eval_out/${alg}"
mkdir -p "./eval_out/${alg}/${dataset}"

if [ ! -d "./eval_out/${alg}" ];
then
  mkdir "./eval_out/${alg}"
fi

log="./eval_out/${alg}/eval.log"

for file in $files
do
  echo $file
  python style_acc/evaluator_roberta.py --dataset ${dataset} --algorithm ${alg} --outdir outputs/${alg}/${dataset}/${file}
  if [ $? -eq 0 ]
  then
    echo "${file} evaluator_roberta.py done" >> ${log}
  else
    echo "${file} evaluator_roberta.py failed" >> ${log}
  fi
  python content_preservation/evaluator_bleu.py --dataset ${dataset} --algorithm ${alg} --outdir outputs/${alg}/${dataset}/${file} --hasref --multiref
  if [ $? -eq 0 ]
  then
    echo "${file} evaluator_bleu.py done" >> ${log}
  else
    echo "${file} evaluator_bleu.py failed" >> ${log}
  fi
  python fluency/evaluator_ppl_kenlm.py --dataset ${dataset} --algorithm ${alg} --outdir outputs/${alg}/${dataset}/${file}
  if [ $? -eq 0 ]
  then
    echo "${file} evaluator_ppl_kenlm.py done" >> ${log}
  else
    echo "${file} evaluator_ppl_kenlm.py failed" >> ${log}
  fi
done

python list2csv.py --algorithm ${alg} --dataset ${dataset}
if [ $? -eq 0 ]
then
  echo "${file} list2csv.py done" >> ${log}
else
  echo "${file} list2csv.py failed" >> ${log}
fi