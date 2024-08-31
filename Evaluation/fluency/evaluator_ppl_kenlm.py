import os, sys
import math
import torch
import argparse
sys.path.append(".")
from utils.dataset import LABEL_MAP, read_file, process_text
from utils.setting import DATA_DIR, BASE_DIR
from nltk import word_tokenize
from pathlib import Path
curPath = Path(__file__)
import kenlm

class Evaluator:
    def __init__(self, args, device="cpu"):
        self.dataset_name = args.dataset
        self.device = device
        # kenlm model for ppl
        self.lm_model = []
        # for label in LABEL_MAP[self.dataset_name]:
        lm_dir = os.path.join(curPath.parent, "lm", self.dataset_name, "kenlm")
        for style in ["0", "1"]:
            lm_path = f'{lm_dir}/ppl_{style}.bin'
            self.lm_model.append(kenlm.LanguageModel(lm_path))

    def compute_ppl_kenlm(self, sents, labels):
        total_score = 0
        total_length = 0
        for sent, label in zip(sents, labels):
            total_score += self.lm_model[1-label].score(sent)
            total_length += len(sent.split())

        if total_length == 0:
            print(total_score, total_length)
            return math.pow(10, 4)
        else:
            return math.pow(10, -total_score / (total_length))
        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl.item()

    def evaluate_file(self, result_file, args):
        transfered_sents = []
        labels = []
        i = int(result_file[-1])
        # sents = read_file(result_file)
        # transfered_sents += sent
        # ppl的处理需要谨慎，当结果是plm输出时，标点和token连在一起，要和kenlm一样分词处理
        data = []
        with open(result_file, 'r', encoding='utf8') as f:
            for line in f:
                pre_data = process_text(line)
                slist = [''.join(word) for word in word_tokenize(pre_data)]
                s = ' '.join(slist)
                data.append(s)
        transfered_sents += data
        labels += [i] * len(data)

        ppl = self.compute_ppl_kenlm(transfered_sents, labels)

        return ppl


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="gyafc_fr")
    parser.add_argument('--algorithm', type=str, default="bsrr")
    parser.add_argument('--outdir', type=str, default="outputs/bsrr/gyafc_fr/decay_rate:0.8_cyclic:1.0_style:1.0_epoch:0")
    parser.add_argument('--file', type=str, default="all",help='')
    parser.add_argument('--cpu', action='store', default=False)
    args = parser.parse_args()

    if not os.path.exists(f'{BASE_DIR}/eval_out/{args.algorithm}'):
        os.mkdir(f'{BASE_DIR}/eval_out/{args.algorithm}')

    if args.file == "all":
        result_files = set()
        for file in os.listdir(f'{BASE_DIR}/{args.outdir}'):
            file_split = file.split(".")
            if len(file_split) == 2 and file_split[0] != "log":
                result_files.add(os.path.join(BASE_DIR, args.outdir, file))
        result_files = sorted(list(result_files))
    else:
        result_files = [args.file]

    device = "cuda" if not args.cpu and torch.cuda.is_available() else "cpu"
    evaluator = Evaluator(args, device=device)
    for result_file in result_files:
        # result_file = f'{BASE_DIR}/data/yelp/test.0'
        ppl = evaluator.evaluate_file(result_file, args)
        eval_path = f'{BASE_DIR}/eval_out/{args.algorithm}/{args.dataset}/{result_file.split("/")[-2]}'
        if not os.path.exists(eval_path):
            os.mkdir(eval_path)
        with open(f'{eval_path}/gen{result_file.split("/")[-1].split(".")[1]}.txt', 'a') as fin:
            fin.write(f'{ppl}\n')

