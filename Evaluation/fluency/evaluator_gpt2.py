# -*- coding: utf-8 -*-
import os
import torch.nn as nn
from torch import cuda
import torch
import argparse
import numpy as np
from transformers import logging
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel

from train_ppl_gpt2 import paired_collate_fn, GPT2Dataset

logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from utils.setting import BASE_DIR, DATA_DIR
from utils.dataset import LABEL_MAP, read_file

def read_insts(file, tokenizer):
    seqs = []
    with open(file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            line = line.lower()
            seq_id = tokenizer.encode(line.strip()[:80])
            seqs.append(seq_id)
    del tokenizer

    return seqs

class Evaluator:
    def __init__(self, args, device):

        self.cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_name = args.dataset
        self.data_dir = DATA_DIR
        self.device = device

    def __get_data__(self, file_name="test"):
        data = []
        for label in LABEL_MAP[self.dataset_name]:
            if "gyafc" in self.dataset_name and file_name == "reference":
                gyafc_ref = [read_file(os.path.join(self.data_dir, "{}.{}.{}".format(file_name, label, i))) for i in
                             range(4)]
                label_data = list(zip(gyafc_ref[0], gyafc_ref[1], gyafc_ref[2], gyafc_ref[3]))
                data += [[sent.split() for sent in sents] for sents in label_data]
            else:
                label_data = read_file(os.path.join(self.data_dir, "{}.{}".format(file_name, label)))
                data += [[sent.split()] for sent in label_data]
        return data


    def cal_ppl(self, model, test_loader, loss_fn):
        ppl_all = []
        loss_list = []
        with torch.no_grad():
            for batch in test_loader:
                src, tgt = map(lambda x: x.to(self.device), batch)
                logits = model(src)[0]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = tgt[..., 1:].contiguous()
                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                               shift_labels.view(-1))

                ppl_batch = torch.exp(loss).cpu().tolist()
                ppl_all.append(ppl_batch)
                loss_list.append(loss.item())

        ppl = sum(ppl_all) / len(ppl_all)
        print('[Info] test | loss {:.4f} | ppl {:.4f}'.format(np.mean(loss_list), ppl))
        # with open('lai.txt', 'w') as f1:
        #     for line, ppl in zip(seqs, ppl_all):
        #         line = line.strip() + str(round(ppl, 6)) + '\n'
        #         print(line)
        #         f1.write(line)
        return ppl

    def evaluate_file(self, result_file, args):
        label = 1-int(result_file[-1])
        torch.manual_seed(args.seed)
        special_tokens = [{'bos_token': '<bos>'}]
        tokenizer = GPT2Tokenizer.from_pretrained('/home/xyf/LanguageModel/gpt2')
        for x in special_tokens:
            tokenizer.add_special_tokens(x)

        model = GPT2LMHeadModel.from_pretrained('/home/xyf/LanguageModel/gpt2')
        model.resize_token_embeddings(len(tokenizer))
        # model.load_state_dict(torch.load(f'{BASE_DIR}/fluency/lm/{args.dataset}/gpt2/gpt2_{label}.chkpt'))
        model.load_state_dict(torch.load(f'{BASE_DIR}/fluency/lm/gyafc_lai/gpt2_{label}.chkpt'))
        model.to(self.device).eval()

        test_src = read_insts(result_file, tokenizer)
        test_tgt = test_src.copy()
        test_loader = torch.utils.data.DataLoader(
            GPT2Dataset(
                src_inst=test_src,
                tgt_inst=test_tgt),
            num_workers=2,
            batch_size=args.batch_size,
            collate_fn=paired_collate_fn)

        loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.eos_token_id)

        print('[Info] {} instances in total.'.format(len(test_src)))
        ppl = self.cal_ppl(model, test_loader, loss_fn)

        return ppl


def main():
    parser = argparse.ArgumentParser('Calculating GPT-2 based perplexity of sentence')
    parser.add_argument('-batch_size', default=32, type=int, help='the size in a batch')
    parser.add_argument('-seed', default=42, type=int, help='pseudo random generator seed')
    parser.add_argument('--dataset', type=str, default="gyafc_fr")
    parser.add_argument('--algorithm', type=str, default="bsrr")
    parser.add_argument('--outdir', type=str, default="fluency/lai")
    parser.add_argument('--file', type=str, default="all",help='')
    parser.add_argument('--cpu', action='store', default=False)

    args = parser.parse_args()
    print('[Info]', args)
    torch.manual_seed(args.seed)
    args.outdir = os.path.join(BASE_DIR, args.outdir)

    if not os.path.exists(f'{BASE_DIR}/eval_out/{args.algorithm}'):
        os.mkdir(f'{BASE_DIR}/eval_out/{args.algorithm}')

    if args.file == "all":
        result_files = set()
        for file in os.listdir(args.outdir):
            file_split = file.split(".")
            if len(file_split) == 2 and file_split[0] != "log":
                result_files.add(os.path.join(BASE_DIR, args.outdir, file))
        result_files = sorted(list(result_files))
    else:
        result_files = [args.file]

    device = "cuda" if not args.cpu and torch.cuda.is_available() else "cpu"
    evaluator = Evaluator(args, device=device)
    for result_file in result_files:
        # result_file = f'{BASE_DIR}/data/gyafc_fr/dev.1'
        ppl = evaluator.evaluate_file(result_file, args)
        eval_path = f'{BASE_DIR}/eval_out/{args.algorithm}/{args.dataset}/{result_file.split("/")[-2]}'
        if not os.path.exists(eval_path):
            os.mkdir(eval_path)
        with open(f'{eval_path}/gen{result_file.split("/")[-1].split(".")[1]}.txt', 'a') as fin:
            fin.write(f'{ppl}\n')



if __name__ == "__main__":
    main()
