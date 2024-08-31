# -*- coding: utf-8 -*-

import os
import time
import argparse
import numpy as np
import math
import torch
import torch.nn as nn
from torch import cuda
from torch.nn.utils import clip_grad_norm_
import sys
import shutil
sys.path.append('..')
from utils.setting import DATA_DIR, BASE_DIR
from transformers import logging
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel

logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = 'cuda' if cuda.is_available() else 'cpu'


def read_insts(dataset, style, prefix, tokenizer):
    file = '{}/{}/{}.{}'.format(DATA_DIR, dataset, prefix,style)

    seqs = []
    with open(file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            line = line.lower()
            seq_id = tokenizer.encode(line.strip()[:80])
            seqs.append(seq_id)
    del tokenizer

    return seqs

def collate_fn(insts, pad_token_id=50256):
    ''' Pad the instance to the max seq length in batch '''

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [pad_token_id] * (max_len - len(inst))
        for inst in insts])
    batch_seq = torch.LongTensor(batch_seq)

    return batch_seq


def paired_collate_fn(insts):
    src_inst, tgt_inst = list(zip(*insts))
    src_inst = collate_fn(src_inst)
    tgt_inst = collate_fn(tgt_inst)

    return src_inst, tgt_inst


class GPT2Dataset(torch.utils.data.Dataset):
    def __init__(self, src_inst=None, tgt_inst=None):
        self._src_inst = src_inst
        self._tgt_inst = tgt_inst

    def __len__(self):
        return len(self._src_inst)

    def __getitem__(self, idx):
        return self._src_inst[idx], self._tgt_inst[idx]


def GPT2Iterator(train_src, train_tgt, valid_src, valid_tgt, args):
    '''Data iterator for fine-tuning GPT-2'''

    train_loader = torch.utils.data.DataLoader(
        GPT2Dataset(
            src_inst=train_src,
            tgt_inst=train_tgt),
        num_workers=12,
        batch_size=args.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        GPT2Dataset(
            src_inst=valid_src,
            tgt_inst=valid_tgt),
        num_workers=2,
        batch_size=args.batch_size,
        collate_fn=paired_collate_fn)

    return train_loader, valid_loader


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, lr, decay_step = 1000, 
                       decay_rate=0.9, cur_step=0):
        self.init_lr = lr
        self.cur_step = cur_step
        self._optimizer = optimizer
        self.decay_rate = decay_rate
        self.decay_step = decay_step

    def step(self):
        '''Step with the inner optimizer'''
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        self.cur_step += 1
        if self.cur_step >= self.decay_step:
            times = int(self.cur_step / self.decay_step)
            lr = self.init_lr * math.pow(self.decay_rate,times)
            if lr < 1e-5:
                lr = 1e-5
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = lr
        else:
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = self.init_lr


def evaluate(model, loss_fn, valid_loader, tokenizer, epoch, step):
    '''Evaluation function for GPT-2'''
    model.eval()
    loss_list = []
    ppl_all = []
    with torch.no_grad():
        for batch in valid_loader:
            src, tgt = map(lambda x: x.to(device), batch)
            logits = model(src)[0]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = tgt[..., 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1))

            ppl_batch = torch.exp(loss).cpu().tolist()
            ppl_all.append(ppl_batch)
            loss_list.append(loss.item())

    ppl = sum(ppl_all)/len(ppl_all)
    model.train()
    print('[Info] eval {:02d}-{:06d} | loss {:.4f} | ppl {:.4f}'.format(
          epoch, step, np.mean(loss_list), ppl))

    return np.mean(loss_list)


def main():
    parser = argparse.ArgumentParser('Fine-tuning GPT-2 for evaluating ppl')
    parser.add_argument('-lr', default=3e-5, type=float, help='initial earning rate')
    parser.add_argument('-epoch', default=30, type=int, help='force stop at 20 epoch')
    parser.add_argument('-acc_steps', default=1, type=int, help='accumulation_steps')
    parser.add_argument('-style', default=1, type=int, help='informal(0) vs formal(1)')
    parser.add_argument('-batch_size', default=32, type=int, help='the size in a batch')
    parser.add_argument('-dataset', default='gyafc', type=str, help='the dataset name')
    parser.add_argument('-patience', default=3, type=int, help='early stopping fine-tune')
    parser.add_argument('-seed', default=42, type=int, help='pseudo random generator seed')
    parser.add_argument('-log_step', default=100, type=int, help='print logs every x step')
    parser.add_argument('-eval_step', default=1000, type=int, help='evaluate every x step')

    args = parser.parse_args()
    print('[Info]', args)
    torch.manual_seed(args.seed)
    special_tokens = [{'bos_token': '<bos>'}]
    tokenizer = GPT2Tokenizer.from_pretrained('/home/xyf/LanguageModel/gpt2')
    for x in special_tokens:
        tokenizer.add_special_tokens(x)

    model = GPT2LMHeadModel.from_pretrained('/home/xyf/LanguageModel/gpt2')
    model.resize_token_embeddings(len(tokenizer))
    model.to(device).train()

    train_src = read_insts(args.dataset, args.style, 'train', tokenizer)
    valid_src = read_insts(args.dataset, args.style, 'dev', tokenizer)
    train_tgt = train_src.copy()
    valid_tgt = valid_src.copy()

    print('[Info] {} instances from train set'.format(len(train_src)))
    print('[Info] {} instances from valid set'.format(len(valid_tgt)))

    train_loader, valid_loader = GPT2Iterator(train_src, train_tgt,
                                              valid_src, valid_tgt, args)

    loss_fn =nn.CrossEntropyLoss(ignore_index=tokenizer.eos_token_id)
    optimizer = ScheduledOptim(
        torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09), args.lr, 500)

    step = 0
    loss_list = []
    start = time.time()
    tab, eval_loss = 0, 1e8
    for epoch in range(args.epoch):
        for batch in train_loader:
            step += 1
            src, tgt = map(lambda x: x.to(device), batch)

            logits = model(src)[0]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = tgt[..., 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1))
            loss_list.append(loss.item())

            loss = loss/args.acc_steps
            loss.backward()

            if step % args.acc_steps == 0:
                clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
                optimizer.step()
                optimizer.zero_grad()

            if step % args.log_step == 0:
                lr = optimizer._optimizer.param_groups[0]['lr']
                print('[Info] {:02d}-{:06d} | loss {:.4f} | '
                      'lr {:.6f} | second {:.1f}'.format(epoch, step,
                      np.mean(loss_list), lr, time.time() - start))
                loss_list = []
                start = time.time()

            if ((len(train_loader) > args.eval_step
                 and step % args.eval_step == 0)
                    or (len(train_loader) < args.eval_step
                        and step % len(train_loader) == 0)):
                valid_loss = evaluate(model, loss_fn, valid_loader, tokenizer, epoch, step)
                if eval_loss >= valid_loss:
                    save_path = f'{BASE_DIR}/fluency/lm/{args.dataset}/gpt2'
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                    torch.save(model.state_dict(), f'{save_path}/gpt2_{args.style}_{step}.chkpt')
                    print('[Info] The checkpoint file has been updated.')
                    shutil.copy(f'{save_path}/gpt2_{args.style}_{step}.chkpt',
                                f'{save_path}/gpt2_{args.style}.chkpt')
                    eval_loss = valid_loss
                    tab = 0
                else:
                    tab += 1
                if tab == args.patience:
                    exit()


if __name__ == "__main__":
    main()
