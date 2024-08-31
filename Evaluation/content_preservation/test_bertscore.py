import os, sys
import torch
import argparse
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
sys.path.append(".")
from utils.dataset import LABEL_MAP, read_file
from utils.setting import DATA_DIR, BASE_DIR

from bert_score import BERTScorer
import re
import numpy as np

from bleurt import score
import numpy as np
from statistics import mean

bert_scorer = BERTScorer(baseline_path='/home/xyf/LanguageModel/bertscore/roberta-large',
                                      model_type='/home/xyf/LanguageModel/bertscore/roberta-large',
                                      device='cuda',
                                      rescale_with_baseline=False,
                                      lang='en',
                                      num_layers=9)

BLEURT_model = score.BleurtScorer('/home/xyf/LanguageModel/bleurt-large-512')


def prepare_data(args):
    labels = LABEL_MAP[args.dataset]
#prepare the training data
    for mode in ["train", "dev", "test"]:
        with open('{}.txt'.format(mode),'w') as f3:
            for label in labels:
                with open(os.path.join(args.data_dir, "{}.{}".format(mode, label)),'r') as f1:
                    f3.writelines(f1.readlines())


class Evaluator:
    def __init__(self, args, device="cpu"):
        self.dataset = args.dataset
        self.data_dir = args.datadir
        self.device = device


    def get_data(self, label, file_name="test"):
        data = []
        label_data = read_file(os.path.join(self.data_dir, "{}.{}".format(file_name, label)))
        data += [[word_tokenize(sent)] for sent in label_data]
        return data

    def adding_multiple_references(self, label):
        human_reference_path = f"{self.data_dir}/references/"
        multi_candidate_list = []
        if self.dataset in ["yelp"]:
            ref_num = 4
            file_list_0 = ["reference0.0", "reference1.0", "reference2.0", "reference3.0"]
            file_list_1 = ["reference0.1", "reference1.1", "reference2.1", "reference3.1"]
        elif self.dataset in ["gyafc_fr", "gyafc_em"]:
            ref_num = 4
            file_list_0 = ["ref.formal.0", "ref.formal.1", "ref.formal.2", "ref.formal.3"]
            file_list_1 = ["ref.informal.0", "ref.informal.1", "ref.informal.2", "ref.informal.3"]

        if label==0:
            fp_list = [open(human_reference_path + str(i), encoding="utf-8").readlines() for i in file_list_0]
            for i in range(len(fp_list[0])):
                multi_candidate_list.append([fp_list[j][i].lower() for j in range(ref_num)])
        else:
            fp_list = [open(human_reference_path + str(i), encoding="utf-8").readlines() for i in file_list_1]
            for i in range(len(fp_list[0])):
                multi_candidate_list.append([fp_list[j][i].lower() for j in range(ref_num)])

        return multi_candidate_list


    def get_bert_score(self, output, ori_data):
        try:
            assert len(output) == len(ori_data)
        except:
            print(len(output))

        bertscore_f1s = bert_scorer.score(ori_data, output)[2]
        bertscore_f1s = np.array([max(b, 0) for b in bertscore_f1s.tolist()])
        bertscore = round(bertscore_f1s.mean(), 3)
        return bertscore

    def get_bert_multi(self, transfered_sents, multi_ref):
        score_list = []
        for i, sent in enumerate(transfered_sents):
            refs = multi_ref[i]
            bertscore_f1s = bert_scorer.score(refs, [sent]*len(refs))[2]
            bertscore_f1s = np.array([max(b, 0) for b in bertscore_f1s.tolist()])
            score_list.append(bertscore_f1s.mean())

        return round(sum(score_list)/len(score_list), 3)

    def evaluate_file(self, result_file, hasref, multiref, style):
        ref_bleu = None
        multi_rfbl = None
        transfered_sents = []
        labels = []

        i = int(style)
        sents = read_file(result_file)
        transfered_sents += sents
        labels += [i] * len(sents)
        ori_data = read_file(os.path.join(self.data_dir, "{}.{}".format("test", i)))

        self_bert = self.get_bert_score(transfered_sents, ori_data)

        if hasref:
            ref_data = read_file(os.path.join(self.data_dir, "{}.{}".format("reference", i)))
            ref_bert = self.get_bert_score(transfered_sents, ref_data)
        if multiref:
            multi_ref = self.adding_multiple_references(label=i)
            multi_rfb = self.get_bert_multi(transfered_sents, multi_ref)

        return self_bert, ref_bert, multi_rfb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="gyafc_em")
    parser.add_argument('--algorithm', type=str, default="chatgpt_fs")
    parser.add_argument('--datadir', type=str, default="./data/gyafc_em")
    parser.add_argument('--outdir', type=str, default="/sota_gpt4/sota_files/gyafc_em_0/ctat")
    parser.add_argument('--cpu', action='store', default=False)
    parser.add_argument('--file', type=str, default="all", help='')
    parser.add_argument('--hasref', action='store_true', default=False)
    parser.add_argument('--multiref', action='store_true', default=False)
    args = parser.parse_args()

    args.datadir = f'{DATA_DIR}/{args.dataset}'
    if not os.path.exists(f'{BASE_DIR}/eval_out/{args.algorithm}'):
        os.mkdir(f'{BASE_DIR}/eval_out/{args.algorithm}')


    if args.dataset in ["yelp", "gyafc_fr", "gyafc_em"]:
        args.hasref = True
        args.multiref = True
    elif args.dataset in ["shakespeare", "amazon", "styleptb_ARR", "styleptb_TFU"]:
        args.hasref = True
        args.multiref = False

    device = "cuda" if not args.cpu and torch.cuda.is_available() else "cpu"
    evaluator = Evaluator(args, device=device)

    cand = ["apple."]
    ref = ["I have a pen."]
    bertscore_f1s = bert_scorer.score(ref, cand)[2]
    bertscore_f1s = np.array([max(b, 0) for b in bertscore_f1s.tolist()])
    bertscore = round(bertscore_f1s.mean(), 3)

    bleurt_score = round(mean(BLEURT_model.score(references=ref, candidates=cand, batch_size=4)), 2)

    result_file = '/home/xyf/PycharmProjects/BENCHMARKforTST/BenchmarkEval'+args.outdir
        # print(result_file, end="\t")
    style = '0'
    self_bert, ref_bert, multibert = evaluator.evaluate_file(result_file, hasref=args.hasref, multiref=args.multiref, style=style)
    eval_path = f'{BASE_DIR}/eval_out/{args.algorithm}/{args.dataset}/{result_file.split("/")[-2]}'
    if not os.path.exists(eval_path):
        os.mkdir(eval_path)
    with open(f'{eval_path}/gen{result_file.split("/")[-1].split(".")[1]}.txt', 'a') as fin:
        fin.write(f'{self_bert}\n{ref_bert}\n{multibert}\n')
    # print(f'self_bleu:{self_bleu}, ref_bleu:{ref_bleu}\t')
