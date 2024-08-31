
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

KENLM=/home/xyf/Downloads/kenlm/build/bin

mkdir -p ./lm/$1/kenlm
dir=$(dirname "$PWD")

echo $dir
for file in $(ls $dir/data/$1/)
do
    if [[ $file == train.* ]]; then
        style=${file##*.}
        echo $style
        data=../$cur_path/data/$1/train.$style
        dst=./lm/$1/kenlm/arpa.$style
        datbin=./lm/$1/kenlm/ppl_$style.bin
        $KENLM/lmplz -o 5 --text $data --arpa $dst --discount_fallback
        $KENLM/build_binary $dst $datbin
        rm -f $dst
    fi
done