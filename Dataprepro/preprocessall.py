import re
import os
import argparse
from nltk import word_tokenize
import numpy as np


def amazon_ref_split(file_path):
    for style in ["0", "1"]:
        with open(f'{file_path}/reference.{style}', 'r') as f1, \
                open(f'{file_path}/reference0.{style}', 'w') as f2, \
                open(f'{file_path}/reference1.{style}', 'w') as f3:
            for s in f1.readlines():
                s = s.strip()
                slist = s.split('\t')
                f2.write(slist[0]+'\n')
                f3.write(slist[1]+'\n')


def count_length(file_dir):
    for file in os.listdir(file_dir):
        for mode in ["train.", "dev.", "test.", "ref."]:
            if mode not in file:
                continue
            file_path = os.path.join(file_dir, file)
            length = []
    # file_path = file_dir + '/train.informal'
    # length = []
        extra = []
        count = 0
        with open(file_path, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip().lower()
                line = word_tokenize(line)
                if len(line)>60:
                    # print(line)
                    count+=1
                    extra.append(len(line))
                length.append(len(line))
        print(f'{file_path}_max:{np.max(length)}')
        print(f'{file_path}_mean:{np.mean(length)}')
        print(f'{file_path}_len>45:{count}')
        # print(f'{file_path}_extramean:{np.mean(extra)}')


def yelp_pre(data0, data_processed):
    with open(data0,'r') as f1, open(data_processed, 'w') as f2:
        for s in f1.readlines():
            s = s.lower()
            s = s.strip()
            s = re.sub(r'(\s+)([.,!?;\'])', r'\2', s)
            s = re.sub(r'(\s{2,})',r' ',s)
            s = re.sub(r'(\s)(n\'t)', r'\2', s)
            s = re.sub('\$ \_', r'$_', s)
            s = re.sub('(\( )(.*?)( \))', r'(\2)', s)
            s = re.sub('(``)( )*(.*?)', r"``\3", s)
            s = re.sub('(.*?)( )*(\'\')', r"\1''", s)
            s = re.sub(r'\n*$','',s)
            f2.write(s+'\n')

def amazon_pre(data0, data_processed):
    with open(data0,'r') as f1, open(data_processed, 'w') as f2:
        for s in f1.readlines():
            s = s.lower()
            s = s.strip()
            s = re.sub(r'(\s+)([.,!?;\'])', r'\2', s)
            s = re.sub(r'(\s{2,})',r' ',s)
            s = re.sub(r'(\s)(n\'t)', r'\2', s)
            s = re.sub(r'(\s)t(\s)', '\'t ', s)
            s = re.sub('\$ \_', r'$_', s)
            s = re.sub('(\( )(.*?)( \))', r'(\2)', s)
            s = re.sub('(``)( )*(.*?)', r"``\3", s)
            s = re.sub('(.*?)( )*(\'\')', r"\1''", s)
            s = re.sub(r'\n*$','',s)
            f2.write(s+'\n')

def gyafc_pre(data0, data_processed):
    with open(data0,'r') as f1, open(data_processed, 'w') as f2:
        for s in f1.readlines():
            s = s.lower()
            s = s.strip()
            token_list = word_tokenize(s)
            if len(token_list)>60:
                token_list = token_list[:60]
            s = ' '.join(token_list)
            s = re.sub(r'(\s{2,})',r' ',s)
            s = re.sub(r'\n*$','',s)
            f2.write(s+'\n')


def preprocess_input(data0):
    with open(data0,'r') as f1:
        for s in f1.readlines():
            s = s.strip()
            text = re.sub('\s{2,}', ' ', s)
            text = re.sub('(.*?)( )([\.,!?\'])', r'\1\3', text)
            text = re.sub('([a-z])( )(n\'t)', r'\1\3', text)
            text = re.sub('\$ \_', r'$_', text)
            text = re.sub('(\( )(.*?)( \))', r'(\2)', text)
            text = re.sub('(``)( )*(.*?)', r"``\3", text)
            text = re.sub('(.*?)( )*(\'\')', r"\1''", text)

def pre_no_plm(data0, data_processed):
    with open(data0,'r') as f1, open(data_processed, 'w') as f2:
        for s in f1.readlines():
            s = s.lower()
            s = s.strip()
            slist = [''.join(word) for word in word_tokenize(s)]
            if len(slist)>60:
                slist = slist[:60]
            s = ' '.join(slist)
            s = re.sub(r'(\s{2,})',r' ',s)
            s = re.sub(r'\n*$','',s)
            f2.write(s+'\n')


def main():
    parser = argparse.ArgumentParser()
    # Parser arguments
    parser.add_argument('--dataset', type=str, default='yelp')
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--no_plm', action='store_true', help='True')
    args = parser.parse_args()

    file_dir = 'datasets/' + args.dataset
    processed_dir = f'datasets/{args.dataset}_clean'

    # count_length(file_dir)
    if not os.path.exists(processed_dir):
        os.mkdir(processed_dir)
    for file in os.listdir(file_dir):
        for mode in ["train.", "dev.", "test.", "ref.", "reference"]:
            if mode not in file:
                continue
            file_path = os.path.join(file_dir, file)
            processed_path = os.path.join(processed_dir, file)
            if args.dataset in ['yelp'] :
                yelp_pre(file_path, processed_path)
                # preprocess_input(file_path)
            elif args.dataset in ["gyafc_fr", "gyafc_em", "shakespeare", "amazon", "styleptb_ARR", "styleptb_TFU"]:
                if args.no_plm:
                    processed_dir = f'datasets/{args.dataset}_clean_noplm'
                    processed_path = os.path.join(processed_dir, file)
                    if not os.path.exists(processed_dir):
                        os.mkdir(processed_dir)
                    pre_no_plm(file_path, processed_path)
                else:
                    gyafc_pre(file_path, processed_path)



if __name__ == "__main__":
    main()

