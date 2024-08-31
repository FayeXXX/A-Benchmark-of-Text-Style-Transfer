# -*- coding: utf-8 -*-

import random
random.seed(1024)
import numpy as np
import re
import torch
import torch.utils.data
import os
from utils.setting import DATA_DIR

LABEL_MAP = {
    "gyafc_fr": ["informal", "formal"],
    "gyafc_em": ["informal", "formal"],
    "yelp": ["0", "1"],
    "amazon": ["0", "1"],
    "shakespeare": ["0", "1"],
}


def process_text(input_str):
    input_str = input_str.lower()    # 转小写
    input_str = re.sub("\s+", " ", input_str.replace("\t", " ")).strip()    # 去空格
    return input_str


def get_dataset(dataset):

    data_path = os.path.join(DATA_DIR, dataset)
    print('data_path',data_path)
    if dataset in ["yelp", "amazon", "shakespeare", "styleptb_ARR", "styleptb_TFU"]:

        tmp = open(data_path + '/train.0', 'r').readlines()
        train_0 = [[process_text(i), 0] for i in tmp]
        tmp = open(data_path + '/train.1', 'r').readlines()
        train_1 = [[process_text(i), 1] for i in tmp]

        tmp = open(data_path + '/test.0', 'r').readlines()
        valid_0 = [[process_text(i), 0] for i in tmp]
        tmp = open(data_path + '/test.1', 'r').readlines()
        valid_1 = [[process_text(i), 1] for i in tmp]

    elif dataset in ["gyafc_fr", "gyafc_em"]:

        tmp_list_0 = open(data_path + '/train.informal', encoding="utf-8").readlines()
        train_0 = [[process_text(i), 0] for i in tmp_list_0]
        tmp_list_1 = open(data_path + '/train.formal', encoding="utf-8").readlines()
        train_1 = [[process_text(i), 1] for i in tmp_list_1]

        tmp_list_0 = open(data_path + '/dev.informal', encoding="utf-8").readlines()
        valid_0 = [[process_text(i), 0] for i in tmp_list_0]
        tmp_list_1 = open(data_path + '/dev.formal', encoding="utf-8").readlines()
        valid_1 = [[process_text(i), 1] for i in tmp_list_1]

    print('[Info] {} instances from train_0 set'.format(len(train_0)))
    print('[Info] {} instances from train_1 set'.format(len(train_1)))

    train_set = train_0 + train_1
    random.seed(100)
    random.shuffle(train_set)
    valid_set = valid_0 + valid_1
    random.seed(100)
    random.shuffle(valid_set)

    return train_set, valid_set


def read_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            data.append(process_text(line))
    return data


def read_data(dataset, style, max_len, prefix,
              tokenizer, domain=0, ratio=1.0):

    if domain!=0:
        domain = tokenizer.encode(domain, add_special_tokens=False)[0]

    if style == 0:
        src_file = './data/{}/{}.0'.format(dataset, prefix)
        tgt_file = './data/{}/{}.1'.format(dataset, prefix)
    else:
        src_file = './data/{}/{}.1'.format(dataset, prefix)
        tgt_file = './data/{}/{}.0'.format(dataset, prefix)

    src_seq, tgt_seq = [], []
    with open(src_file, 'r') as f1, open(tgt_file, 'r') as f2:
        f1 = f1.readlines()
        f2 = f2.readlines()
        index = [i for i in range(len(f1))]
        random.shuffle(index)
        index = index[:int(len(index) * ratio)]
        for i, (s, t) in enumerate(zip(f1, f2)):
            if i in index:
                s = tokenizer.encode(s)
                t = tokenizer.encode(t)
                s = s[:min(len(s) - 1, max_len)] + s[-1:]
                t = t[:min(len(t) - 1, max_len)] + t[-1:]
                s[0] = domain
                src_seq.append(s)
                tgt_seq.append([tokenizer.bos_token_id]+t)

    return src_seq, tgt_seq


def collate_fn(insts, pad_token_id=1):
    ''' Pad the instance to the max seq length in batch '''

    max_len = max(len(inst) for inst in insts)
    max_len = max_len if max_len > 4 else 5

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


class CNNDataset(torch.utils.data.Dataset):
    def __init__(self, insts, label):
        self.insts = insts
        self.label = label

    def __getitem__(self, index):
        return self.insts[index], self.label[index]

    def __len__(self):
        return len(self.insts)


def SCIterator(insts_0, insts_1, opt, pad_token_id=1, shuffle=True):
    '''Data iterator for style classifier'''

    def cls_fn(insts):
        insts, labels = list(zip(*insts))
        seq = collate_fn(insts, pad_token_id)
        labels = torch.LongTensor(labels)
        return (seq, labels)

    num = len(insts_0) + len(insts_1)
    loader = torch.utils.data.DataLoader(
        CNNDataset(
            insts=insts_0 + insts_1,
            label=[0 if i < len(insts_0)
                   else 1 for i in range(num)]),
        shuffle=shuffle,
        num_workers=2,
        collate_fn=cls_fn,
        batch_size=opt.batch_size)

    return loader


def load_embedding(tokenizer, embed_dim, embed_path=None):
    '''Parse an embedding text file into an array.'''

    embedding = np.random.normal(scale=embed_dim ** -0.5,
                                 size=(len(tokenizer), embed_dim))
    if embed_path == None:
        return embedding

    print('[Info] Loading embedding')
    embed_dict = {}
    with open(embed_path) as file:
        for i, line in enumerate(file):
            if i == 0:
                continue
            tokens = line.rstrip().split()
            try:
                embed_dict[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
            except:
                continue

    for i in range(len(tokenizer)):
        try:
            word = tokenizer.decode(i)
            if word in embed_dict:
                embedding[i] = embed_dict[word]
        except:
            print(i)

    return embedding

