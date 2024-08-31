import torch
import os
import argparse
from transformers import RobertaForSequenceClassification, RobertaTokenizer, logging
logging.set_verbosity_error()
from sklearn.metrics import accuracy_score
import sys
sys.path.append(".")
from utils.dataset import LABEL_MAP, read_file
from utils.setting import DATA_DIR, BASE_DIR
from pathlib import Path
curPath = Path(__file__)


class Evaluator:
    def __init__(self, args, device="gpu"):

        self.dataset_name = args.dataset
        self.device = device

        pretrained_style_model = "/home/xyf/LanguageModel/roberta-base"
        classifier_path = os.path.join(curPath.parent, "classifier", self.dataset_name)

        self.style_classifier = RobertaForSequenceClassification.from_pretrained(pretrained_style_model, output_attentions=True)
        self.style_classifier_tokenizer = RobertaTokenizer.from_pretrained(pretrained_style_model, use_fast=True)
        self.style_classifier_tokenizer.model_max_length = 512
        self.style_classifier.cuda().eval()
        self.style_classifier.load_state_dict(torch.load(classifier_path + '/TextBERT_best.chkpt'))

    def get_acc(self, sents, labels):

        batch = 64
        test_batches = []
        for i in range(len(sents) // batch + bool(len(sents) % batch)):
            test_batches.append(sents[i * batch:(i + 1) * batch])

        style_CLS_pred = []
        for batch in test_batches:
            batch_text = [i.replace("</s>", "").replace("<s>", "") for i in batch]
            batch_text = [i[i.index("#") + 1:] if "#" in i else i for i in batch_text]
            batch_input = self.style_classifier_tokenizer(batch_text, add_special_tokens=True, padding=True, truncation=True,
                                                          return_tensors="pt").data
            logits = self.style_classifier(batch_input["input_ids"].cuda(),
                                       attention_mask=batch_input["attention_mask"].cuda()).logits.detach()

            prediction = torch.argmax(logits, dim=-1)
            style_CLS_pred.extend(prediction.detach().cpu().numpy().tolist())

        all_style_cls_gold = labels
        style_accuracy = accuracy_score(y_pred=style_CLS_pred, y_true=all_style_cls_gold)

        return style_accuracy

    def evaluate_file(self, result_file):
        transfered_sents = []
        labels = []

        i = int(result_file[-1])
        sents = read_file(result_file)
        transfered_sents += sents
        labels += [int(not(i))] * len(sents)

        acc_bert = self.get_acc(transfered_sents, labels)
        return acc_bert

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="amazon")
    parser.add_argument('--algorithm', type=str, default="chagpt_fs")
    parser.add_argument('--outdir', type=str, default="outputs/chatgpt_fs/shakespeare/gpt-3.5-turbo-1106")
    parser.add_argument('--cpu', action='store', default=False)
    parser.add_argument('--file', type=str, default="all", help='')
    args = parser.parse_args()

    if not os.path.exists(f'{BASE_DIR}/eval_out/{args.algorithm}'):
        os.mkdir(f'{BASE_DIR}/eval_out/{args.algorithm}')

    if args.file == "all":
        result_files = set()
        # for file in os.listdir(f'{BASE_DIR}/{args.outdir}'):
        for file in os.listdir(f'{args.outdir}'):
            file_split = file.split(".")
            if len(file_split) == 2 and file_split[0] != "log":
                result_files.add(os.path.join(BASE_DIR, args.outdir, file))
        result_files = sorted(list(result_files))
    else:
        result_files = [args.file]

    device = "cuda" if not args.cpu and torch.cuda.is_available() else "cpu"
    evaluator = Evaluator(args, device=device)
    for result_file in result_files:
        acc_bert = evaluator.evaluate_file(result_file)
        eval_path = f'{BASE_DIR}/eval_out/{args.algorithm}/{args.dataset}/{result_file.split("/")[-2]}'
        if not os.path.exists(eval_path):
            os.mkdir(eval_path)
        with open(f'{eval_path}/gen{result_file.split("/")[-1].split(".")[1]}.txt', 'a') as fin:
            fin.write(f'{acc_bert}\n')