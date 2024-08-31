from torch.utils.data import Dataset
import numpy as np
from transformers import pipeline
import os,sys
from torch import cuda
import torch
import argparse
from transformers import logging
logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = 'cuda' if cuda.is_available() else 'cpu'
sys.path.append(".")
from utils.setting import BASE_DIR
from utils.dataset import read_file
import re

class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)


class Evaluator:
    def __init__(self, args, device="cpu"):
        self.cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.device = device

    def cal_ppl(self, args, transfered_sents):
        # 去掉多余的空格，否则影响cola分类的判断
        # sent_clean = []
        # for sent in transfered_sents:
        #     text = re.sub('\s{2,}', ' ', sent)  # 2个以上的空替换成1个
        #     text = re.sub('(.*?)( )([\.,!?\'])', r'\1\3', text)  # 遇空格+(\.,!?'符号)将空格删除
        #     text = re.sub('([a-z])( )(n\'t)', r'\1\3', text)  # 遇字母+空格+n't 将空格删除
        #     sent_clean.append(text)
        output_dataset = ListDataset(transfered_sents)
        fluency_corrects = []
        fluency_label = 'LABEL_0'
        fluency_classifier = pipeline(
            'text-classification',
            model='/home/xyf/LanguageModel/roberta-large-cola-krishna2020',
            device_map='auto')
        for i, c in enumerate(fluency_classifier(output_dataset, batch_size=args.batch_size, truncation=True)):
            fluency_corrects.append(int(c['label'] == fluency_label))
            if not c['label'] == fluency_label:
                print(f'{i}:{transfered_sents[i]}\n')
        fluency_corrects = np.array(fluency_corrects)
        fluency = round(100 * fluency_corrects.mean(), 1)
        return fluency

    def evaluate_file(self, result_file, args):
        transfered_sents = []
        sents = read_file(result_file)
        transfered_sents += sents
        print('[Info] {} instances in total.'.format(len(transfered_sents)))
        ppl = self.cal_ppl(args, transfered_sents)
        return ppl

def main():
    parser = argparse.ArgumentParser('Calculating GPT-2 based perplexity of sentence')
    parser.add_argument('-batch_size', default=32, type=int, help='the size in a batch')
    parser.add_argument('-seed', default=42, type=int, help='pseudo random generator seed')
    parser.add_argument('--dataset', type=str, default="amazon")
    parser.add_argument('--algorithm', type=str, default="bsrr")
    parser.add_argument('--outdir', type=str, default="data/amazon/test.1")
    parser.add_argument('--file', type=str, default="all",help='')
    parser.add_argument('--cpu', action='store', default=False)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if not os.path.exists(f'{BASE_DIR}/eval_out/{args.algorithm}'):
        os.mkdir(f'{BASE_DIR}/eval_out/{args.algorithm}')

    # if args.file == "all":
    #     result_files = set()
    #     for file in os.listdir(f'{BASE_DIR}/{args.outdir}'):
    #         file_split = file.split(".")
    #         if len(file_split) == 2 and file_split[0] != "log":
    #             result_files.add(os.path.join(BASE_DIR, args.outdir, file))
    #     result_files = sorted(list(result_files))
    # else:
    #     result_files = [args.file]

    device = "cuda" if not args.cpu and torch.cuda.is_available() else "cpu"
    evaluator = Evaluator(args, device=device)
    ppl = evaluator.evaluate_file(os.path.join(BASE_DIR, args.outdir), args)
    # for result_file in result_files:
    #     ppl = evaluator.evaluate_file(result_file, args)
    #     eval_path = f'{BASE_DIR}/eval_out/{args.algorithm}/{args.dataset}/{result_file.split("/")[-2]}'
    #     if not os.path.exists(eval_path):
    #         os.mkdir(eval_path)
    #     with open(f'{eval_path}/gen{result_file.split("/")[-1].split(".")[1]}.txt', 'a') as fin:
    #         fin.write(f'{ppl}\n')

if __name__ == "__main__":
    main()



