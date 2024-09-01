# -*- coding: utf-8 -*-

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import argparse

import torch
from torch import cuda
from model import BartModel
from model import BartForMaskedLM
from transformers import BartTokenizer

device = 'cuda' if cuda.is_available() else 'cpu'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("-p", type=float, default=0.9)
    parser.add_argument("-length", type=int, default=30)
    parser.add_argument('-order', default=0, type=str, help='order')
    parser.add_argument('-model', default='bart', type=str, help='model')
    parser.add_argument("-seed", type=int, default=42, help="random seed")
    parser.add_argument('-style', default=0, type=int, help='from 0 to 1')
    parser.add_argument('-dataset', default='em', type=str, help='dataset')
    parser.add_argument('-bl_factor', default=0.2, type=float, help='proportion of data')
    parser.add_argument('-sc_factor', default=1., type=float, help='proportion of data')

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    tokenizer = BartTokenizer.from_pretrained("/home/xyf/LanguageModel/bart-large")
    # for token in ['<E>', '<F>']:
    #     tokenizer.add_tokens(token)
    # if opt.dataset == 'em':
    #     domain = tokenizer.encode('<E>', add_special_tokens=False)[0]
    # else:
    #     domain = tokenizer.encode('<F>', add_special_tokens=False)[0]
    model = BartModel.from_pretrained("/home/xyf/LanguageModel/bart-large")
    model.config.output_past = True
    model = BartForMaskedLM.from_pretrained("/home/xyf/LanguageModel/bart-large",
                                            config=model.config)
    model.to(device).eval()
    model.load_state_dict(torch.load('checkpoints/{}_{}/sc:{}_bl:{}_{}.chkpt'.format(
        opt.model, opt.dataset, opt.sc_factor, opt.bl_factor, opt.style)))
    print('Loading:checkpoints/{}_{}/sc:{}_bl:{}_{}.chkpt'.format(
        opt.model, opt.dataset, opt.sc_factor, opt.bl_factor, opt.style))

    src_seq = []
    with open('./data/{}/test.{}'.format(opt.dataset, opt.style)) as fin:
        for line in fin.readlines():
            src_seq.append(line.strip())

    start = time.time()
    if not os.path.exists('./data/outputs/{}_{}'.format(opt.model, opt.dataset)):
        os.mkdir('./data/outputs/{}_{}'.format(opt.model, opt.dataset))
    with open('./data/outputs/{}_{}/sc:{}_bl:{}.{}.txt'.format(opt.model, opt.dataset, opt.sc_factor, opt.bl_factor, opt.style), 'w') as fout:
        for idx, line in enumerate(src_seq):
            if idx % 100 == 0:
                print('[Info] processing {} seqs | sencods {:.4f}'.format(
                    idx, time.time() - start))
                start = time.time()
            src = tokenizer.encode(line, return_tensors='pt')
            generated_ids = model.generate(src.to(device),
                                           num_beams=5,
                                           max_length=30)
            text = [tokenizer.decode(g, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
                    for g in generated_ids][0]
            fout.write(text.strip() + '\n')


if __name__ == "__main__":
    main()
