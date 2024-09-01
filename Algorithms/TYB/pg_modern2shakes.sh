#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=50000

#python train_shakes.py -style 0 -ratio 1.0 -dataset $1 -order $2 -sc_factor $3 -bl_factor $4 -$5 -$6
#python infer.py -style 0 -dataset $1 -order $2 -sc_factor $3 -bl_factor $4
#rm checkpoints/bart_$1_$2_0.chkpt
#
python train_shakes.py -style 1 -ratio 1.0 -dataset $1 -order $2 -sc_factor $3 -bl_factor $4 -$5 -$6
python infer.py -style 1 -dataset $1 -order $2 -sc_factor $3 -bl_factor $4
##rm checkpoints/bart_$1_$2_1.chkpt
#
#echo "----------------Style----------------"
#python classifier/test.py -dataset $1 -order $2 -sc_factor $3 -bl_factor $4
#
#echo "----------------BLEU----------------"
#mkdir -p data/$1/outputs/
#mkdir -p data/$1/outputs/bart_$1/
#python utils/tokenizer.py data/outputs/bart_$1/sc:$3_bl:$4.0.txt data/$1/outputs/bart_$1/sc:$3_bl:$4.0.txt False
#python utils/tokenizer.py data/outputs/bart_$1/sc:$3_bl:$4.1.txt data/$1/outputs/bart_$1/sc:$3_bl:$4.1.txt Flase
#
###对reference的tokenize
##python utils/tokenizer_ref.py data/$1/test/ data/$1/original_ref/ False
#perl utils/multi-bleu.perl data/$1/original_ref/$7.ref < data/$1/outputs/bart_$1/sc:$3_bl:$4.0.txt
#perl utils/multi-bleu.perl data/$1/original_ref/$8.ref < data/$1/outputs/bart_$1/sc:$3_bl:$4.1.txt
#
##echo "----------------BLEURT----------------"
#python utils/cal_bleurt.py data/outputs/bart_$1/sc:$3_bl:$4.0.txt data/outputs/bart_$1/sc:$3_bl:$4.1.txt \
#                         data/$1/test/$7.ref data/$1/test/$8.ref

