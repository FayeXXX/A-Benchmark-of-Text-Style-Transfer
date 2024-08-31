import os
import sys
import time
import argparse
import random
import re
import numpy as np
import torch
import torch.nn as nn
from torch import cuda
sys.path.append("..")
from utils.dataset import get_dataset

from transformers import RobertaForSequenceClassification, RobertaTokenizer
device = 'cuda' if cuda.is_available() else 'cpu'

from pathlib import Path
curPath = Path(__file__)

def get_batches(data, batch_size):
    batches = []
    for i in range(len(data) // batch_size + bool(len(data) % batch_size)):
        batches.append(data[i * batch_size:(i + 1) * batch_size])

    return batches


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.determinstic = True


def evaluate_sc(model, valid_loader, loss_fn, epoch, tokenizer):
    """ Evaluation function for style classifier """
    model.eval()
    total_acc = 0.
    total_num = 0.
    total_loss = 0.
    with torch.no_grad():
        for batch in valid_loader:
            x_batch = [i[0] for i in batch]
            x_batch = tokenizer(x_batch, add_special_tokens=True, padding=True, return_tensors="pt").data
            y_batch = torch.tensor([i[1] for i in batch]).to(device)
            logits = model(x_batch["input_ids"].to(device), attention_mask=x_batch["attention_mask"].to(device)).logits

            total_loss += loss_fn(logits, y_batch)
            _, y_hat = torch.max(logits, dim=-1)
            same = [float(p == q) for p, q in zip(y_batch, y_hat)]
            total_acc += sum(same)
            total_num += len(y_batch)
    model.train()
    print('[Info] Epoch {:02d}-valid: {}'.format(
        epoch, 'acc {:.4f}% | loss {:.4f}').format(
        total_acc / total_num * 100, total_loss / total_num))

    return total_acc / total_num, total_loss / total_num


def main():
    parser = argparse.ArgumentParser('Style Classifier TextCNN')
    parser.add_argument('-dataset', default='amazon', type=str, help='dataset name')
    parser.add_argument('-lr', default=2e-5, type=float, help='learning rate')
    parser.add_argument('-seed',  default=100, type=int, help='pseudo random number seed')
    parser.add_argument('-min_count', default=0, type=int, help='minmum number of dataset')
    parser.add_argument('-max_len', default=30, type=int, help='maximum tokens in a batch')
    parser.add_argument('-log_step', default=100, type=int, help='print log every x steps')
    parser.add_argument('-eval_step', default=3000, type=int, help='early stopping training')
    parser.add_argument('-batch_size', default=64, type=int, help='maximum sents in a batch')
    parser.add_argument('-epoch', default=10, type=int, help='force stop at specified epoch')

    args = parser.parse_args()

    setup_seed(args.seed)

    pretrained_style_model = "/home/xyf/LanguageModel/roberta-base"
    tokenizer = RobertaTokenizer.from_pretrained(pretrained_style_model)

    train_set, valid_set = get_dataset(args.dataset)
    train_batches = get_batches(train_set, args.batch_size)
    valid_batches = get_batches(valid_set, args.batch_size)

    model = RobertaForSequenceClassification.from_pretrained(pretrained_style_model)
    model.to(device).train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00002, betas=(0.9, 0.98), eps=1e-09)
    loss_fn = nn.CrossEntropyLoss()

    print('[Info] Built a model with {} parameters'.format(
        sum(p.numel() for p in model.parameters())))
    print('[Info]', args)

    tab = 0
    avg_acc = 0
    total_acc = 0.
    total_num = 0.
    total_loss = 0.
    start = time.time()
    train_steps = 0

    for e in range(args.epoch):

        model.train()
        for idx, batch in enumerate(train_batches):
            x_batch = [i[0] for i in batch]
            x_batch = tokenizer(x_batch, add_special_tokens=True, padding=True, return_tensors="pt").data
            y_batch = torch.tensor([i[1] for i in batch]).to(device)

            train_steps += 1

            optimizer.zero_grad()
            logits = model(x_batch["input_ids"].to(device), attention_mask=x_batch["attention_mask"].to(device)).logits
            loss = loss_fn(logits, y_batch)
            total_loss += loss
            loss.backward()
            optimizer.step()

            y_hat = logits.argmax(dim=-1)
            same = [float(p == q) for p, q in zip(y_batch, y_hat)]
            total_acc += sum(same)
            total_num += len(y_batch)

            if train_steps % args.log_step == 0:
                lr = optimizer.param_groups[0]['lr']
                print('[Info] Epoch {:02d}-{:05d}: | average acc {:.4f}% | '
                      'average loss {:.4f} | lr {:.6f} | second {:.2f}'.format(
                    e, train_steps, total_acc / total_num * 100,
                                    total_loss / (total_num), lr, time.time() - start))
                start = time.time()

            if train_steps % args.eval_step == 0:
                valid_acc, valid_loss = evaluate_sc(model, valid_batches, loss_fn, e, tokenizer)
                # if avg_acc < valid_acc or True:
                if avg_acc < valid_acc:
                    avg_acc = valid_acc
                    save_path = os.path.join(curPath.parent, 'classifier', args.dataset)
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                    # torch.save(model.state_dict(), save_path + '/TextBERT_best.chkpt' + str(train_steps) + "_" + str(valid_acc))
                    torch.save(model.state_dict(), save_path + '/TextBERT_best.chkpt')
                    print('[Info] The checkpoint file has been updated.')
                    tab = 0
                else:
                    tab += 1
                    if tab == 10:
                        exit()


if __name__ == '__main__':
    main()

