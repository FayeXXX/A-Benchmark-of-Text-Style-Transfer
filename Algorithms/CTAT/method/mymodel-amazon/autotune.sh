#!/usr/bin/env sh

log="running.log"

python main.py --smooth ${1} --dropout ${2} --dataset 'amazon' --learning_rate 5e-4
if [ $? -eq 0 ]
then echo "--smooth ${1} --dropout ${2} done" >> ${log}
else echo "--smooth ${1} --dropout ${2} failed" >> ${log}
fi


