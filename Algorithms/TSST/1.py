from nltk.translate.bleu_score import corpus_bleu
import json
from nltk import word_tokenize



score = corpus_bleu([['He is a great singer.']], ['He sings really well.'])
print(score)

score = corpus_bleu([['He is a great singer.']], ['He is a great writer.'])
print(score)


# statistic for OOV
# set_vocab = []
# with open('./data/yelp/vocab.txt',encoding="utf8") as f:
#     for line in f:
#         splitline = line.strip().split('\t')
#         set_vocab.append(splitline[0])
#
# vocab_test = []
# with open('./data/yelp/test.0',encoding="utf8") as f:
#     for line in f:
#         words = line.split()
#         for word in words:
#             vocab_test.append(word)
#
# with open('./data/yelp/test.1',encoding="utf8") as f:
#     for line in f:
#         words = line.split()
#         for word in words:
#             vocab_test.append(word)
#
# set_vocab_test = set(vocab_test)

vocab = []
with open('./data/gyafc/gyafc_vocab.txt','r') as f:
    for line in f:
        splitline = line.strip().split('\t')
        vocab.append(splitline[0])
set_vocab = set(vocab)

vocab_test = []
with open('./data/gyafc/gyafc_unpaired_test.json','r') as f:
    for idx, line in enumerate(f):
        words = json.loads(line)
        for word in words[0]['sent'].split():
            vocab_test.append(word)
set_test = set(vocab_test)

vocab_train = []
with open('./data/gyafc/gyafc_unpaired_train.json','r') as f:
    for idx, line in enumerate(f):
        words = json.loads(line)
        for word in words[0]['sent'].split():
            vocab_train.append(word)

# vocab_test = []
# with open('./data/yelp/yelp_unpaired_test.json','r') as f:
#     for idx, line in enumerate(f):
#         words = json.loads(line)
#         for word in word_tokenize(words['sent']):
#             vocab_test.append(word)
# set_test = set(vocab_test)
#
# vocab_train = []
# with open('./data/yelp/yelp_unpaired_train.json','r') as f:
#     for idx, line in enumerate(f):
#         words = json.loads(line)
#         for word in word_tokenize(words['sent']):
#             vocab_train.append(word)


set_train = set(vocab_train)
ints_test = set_vocab.intersection(set_test)
ints_train = set_vocab.intersection(set_train)
print('vocab_all:',len(set_vocab))
print('vocab_test:',len(set_test))
print('vocab_train:',len(set_train))
print('ints_test:',len(ints_test))
print('ints_train:',len(ints_train))

#in test but not in vocab
test_i = set_test - ints_test
#in train but not in vocab
train_i = set_train - ints_train

print('done')
