# -*- coding: utf-8 -*-
import os
import sys
import nltk

file_dir = os.listdir(sys.argv[1])
for file in file_dir:
    fin = open(sys.argv[1] + file, 'r').readlines()
    new_file = sys.argv[2] + file
    with open(new_file,'w') as f:
        for line in fin:
            if sys.argv[3]=='True':
                line = nltk.word_tokenize(line.strip().lower())
            else:
                line = nltk.word_tokenize(line.strip())
            f.write(' '.join(line)+'\n')
