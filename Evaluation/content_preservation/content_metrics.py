import os
import torch
import argparse
from nltk.translate.bleu_score import corpus_bleu
from bleurt import score
from nltk.tokenize import word_tokenize
import numpy as np
from statistics import mean
from bert_score import BERTScorer
from mutual_implication_score import MIS
from simcse import SimCSE
from sacrebleu.metrics import CHRF
import sys
sys.path.append(".")
from utils.setting import BASE_DIR, DATA_DIR
from utils.dataset import LABEL_MAP,read_file


def prepare_data(args):
    labels = LABEL_MAP[args.dataset]
#prepare the training data
    for mode in ["train", "dev", "test"]:
        with open('{}.txt'.format(mode),'w') as f3:
            for label in labels:
                with open(os.path.join(args.data_dir, "{}.{}".format(mode, label)),'r') as f1:
                    f3.writelines(f1.readlines())


class Evaluator:
    def __init__(self, args, device="cpu", fluency_batch_size=32):

        self.cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_name = args.dataset
        self.data_dir = args.datadir
        self.device = device
        self.bert_scorer = BERTScorer(baseline_path='/home/xyf/LanguageModel/bertscore/roberta-large',
                                      model_type='/home/xyf/LanguageModel/bertscore/roberta-large',
                                      device='cuda',
                                      rescale_with_baseline=False,
                                      lang='en',
                                      num_layers=9)
        self.SimCSE_model = SimCSE("/home/xyf/LanguageModel/sup-simcse-bert-base-uncased")
        self.BLEURT_model = score.BleurtScorer('/home/xyf/LanguageModel/bleurt-large-512')
        self.mis = MIS('/home/xyf/LanguageModel/mis', device='cuda')

    def get_seg_data(self, label, file_name="test"):
        data = []
        if self.dataset_name == "gyafc" and file_name == "reference":
            gyafc_ref = [read_file(os.path.join(self.data_dir, "{}.{}.{}".format(file_name, label, i))) for i in range(4)]
            label_data = list(zip(gyafc_ref[0], gyafc_ref[1], gyafc_ref[2], gyafc_ref[3]))
            data += [[word_tokenize(sents)] for sents in label_data]
        else:
            label_data = read_file(os.path.join(self.data_dir, "{}.{}".format(file_name, label)))
            data += [[word_tokenize(sent)] for sent in label_data]
        return data

    def get_data(self, label, file_name="test"):
        if self.dataset_name == "gyafc" and file_name == "reference":
            gyafc_ref = [read_file(os.path.join(self.data_dir, "{}.{}.{}".format(file_name, label, i))) for i in range(4)]
            data = list(zip(gyafc_ref[0], gyafc_ref[1], gyafc_ref[2], gyafc_ref[3]))
        else:
            data = read_file(os.path.join(self.data_dir, "{}.{}".format(file_name, label)))
        return data

    def get_ref_bleu(self, seg_sents, ref_data):
        try:
            assert len(seg_sents) == len(ref_data)
        except:
            print(len(seg_sents))
        return corpus_bleu(ref_data, seg_sents)

    def get_self_bleu(self, seg_sents, ori_data):
        try:
            assert len(seg_sents) == len(ori_data)
        except:
            print(len(seg_sents))
        return corpus_bleu(ori_data, seg_sents)

    def get_multiple_references(self, label):
        human_reference_path = "../data/yelp/references/"
        multi_ref_data = []
        seg_multi_ref_data = []

        if label==0:
            file_list_0 = ["reference0.0", "reference1.0", "reference2.0", "reference3.0"]

            fp_list = [open(human_reference_path + str(i), encoding="utf-8").readlines() for i in file_list_0]
            for i in range(len(fp_list[0])):
                multi_ref_data.append([fp_list[j][i].lower() for j in range(4)])
                seg_multi_ref_data.append([word_tokenize(fp_list[j][i].lower()) for j in range(4)])
        else:
            file_list_1 = ["reference0.1", "reference1.1", "reference2.1", "reference3.1"]
            fp_list = [open(human_reference_path + str(i), encoding="utf-8").readlines() for i in file_list_1]
            for i in range(len(fp_list[0])):
                multi_ref_data.append([fp_list[j][i].lower() for j in range(4)])
                seg_multi_ref_data.append([word_tokenize(fp_list[j][i].lower()) for j in range(4)])

        return multi_ref_data, seg_multi_ref_data

    # def get_bleurt(self, transfered_sents, ori_data, ref_data):
    #     src = ori_data
    #     src = [' '.join(item[0]) for item in src]
    #     out = transfered_sents
    #     ref = ref_data
    #     if self.dataset_name == "gyafc":
    #         pass
    #     else:
    #         ref = [' '.join(item[0]) for item in ref]
    #
    #     # BLEURT score
    #     self_bleurt = round(mean(BLEURT.score(references=src, candidates=out, batch_size=4)), 4)
    #     ref_bleurt = round(mean(BLEURT.score(references=ref, candidates=out, batch_size=4)), 4)
    #     return self_bleurt, ref_bleurt

    def evaluate_file(self, result_file, hasref=True, multiref=False):
        labels = []
        i = int(result_file[-1])
        transfered_sents = read_file(result_file)
        labels += [i] * len(transfered_sents)
        ori_data = self.get_data(file_name="test", label=i)
        ref_data = self.get_data(file_name="reference",label=i)
        multi_ref, seg_ref_data = self.get_multiple_references(label=i)
        seg_ori_data = [word_tokenize(sent) for sent in ori_data]
        seg_transfered_sents = [word_tokenize(sent) for sent in transfered_sents]

        # if hasref:
        #     ref_data = self.get_data(file_name="reference", label=i)
        #     ref_bleu = self.get_ref_bleu(seg_sents, ref_data)
        # if multiref:
        #     multi_ref = self.adding_multiple_references(label=i)
        #     multi_rfbl = self.get_ref_bleu(seg_sents, multi_ref)

        # chrf https://github.com/mjpost/sacrebleu
        ch_scorer = CHRF()
        chrf_score = ch_scorer.corpus_score(transfered_sents, ref_data)

        # bert_score
        bertscore_f1s = self.bert_scorer.score(transfered_sents, ref_data)[2]
        bertscore_f1s = np.array([max(b, 0) for b in bertscore_f1s.tolist()])
        bertscore = round(bertscore_f1s.mean(), 2)

        #  SIMCSE_score https://github.com/princeton-nlp/SimCSE
        SIMCSE_score_all = []
        for i in range(len(ori_data)):
            SIMCSE_score_all.append(self.SimCSE_model.similarity(transfered_sents[i], ref_data[i]))
        SIMCSE_score = round(np.mean(SIMCSE_score_all),2)

        #  bleurt_score https://github.com/google-research/bleurt
        bleurt_score = round(mean(self.BLEURT_model.score(references=ref_data, candidates=transfered_sents, batch_size=4)), 2)

        #  mis_score https://github.com/s-nlp/mutual_implication_score
        mis_score = round(mean(self.mis.compute(transfered_sents, ref_data)),2)

        return bertscore, SIMCSE_score, bleurt_score, mis_score, chrf_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="yelp")
    parser.add_argument('--algorithm', type=str, default="ctat")
    parser.add_argument('--datadir', type=str, default="./data/yelp")
    parser.add_argument('--outdir', type=str, default="best_results/stytrans")
    parser.add_argument('--cpu', action='store', default=False)
    parser.add_argument('--file', type=str, default="all", help='')
    parser.add_argument('--hasref', action='store_true', default=False)
    parser.add_argument('--multiref', action='store_true', default=False)
    parser.add_argument('--eval_method', type=str, default="bleu", help='')
    args = parser.parse_args()
    print(f"Evaluating algorithm: {args.algorithm}")
    dataset_name = args.dataset
    print(f"Evaluating dataset: {dataset_name}")
    datadir = f'data/{args.dataset}'
    if not os.path.exists(f'{BASE_DIR}/eval_out/{args.algorithm}'):
        os.mkdir(f'{BASE_DIR}/eval_out/{args.algorithm}')

    if args.file == "all":
        result_files = set()
        for file in os.listdir(f'{BASE_DIR}/{args.outdir}'):
            file_split = file.split(".")
            if len(file_split) == 2 and file_split[0] != "log":
                result_files.add(os.path.join(args.outdir, file))
        result_files = sorted(list(result_files))
    else:
        result_files = [args.file]

    device = "cuda" if not args.cpu and torch.cuda.is_available() else "cpu"
    evaluator = Evaluator(args, device=device)
    for result_file in result_files:
        print(result_file, end="\t")
        bertscore, SIMCSE_score, bleurt_score, mis_score, chrf_score = evaluator.evaluate_file(result_file, \
                                                                               hasref=args.hasref, multiref=args.multiref)
        print(f'{bertscore}\n{SIMCSE_score}\n{bleurt_score}\n{mis_score}\n{chrf_score}\n')
        eval_path = f'{BASE_DIR}/eval_out/{args.algorithm}/{result_file.split("/")[-2]}'
        if not os.path.exists(eval_path):
            os.mkdir(eval_path)
        with open(f'{eval_path}/gen{result_file.split("/")[-1].split(".")[1]}.txt', 'a') as fin:
            fin.write(f'{ppl}\n')


