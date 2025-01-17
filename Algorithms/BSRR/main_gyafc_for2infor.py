import os
import random
import time
import re
from collections import Counter
from tqdm import tqdm
import argparse

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize

from utils import get_batches
from Model import Model
import global_config


# os.environ['CUDA_VISIBLE_DEVICES'] = global_config.gpu_id
running_random_number = random.randint(1000, 9999)
print("running_random_number", running_random_number, "\n")

parser = argparse.ArgumentParser()
parser.add_argument('--cyclic', type=float, default=1.0)
parser.add_argument('--style', type=float, default=0.8)
parser.add_argument('--decay_rate', type=float, default=0.6)
parser.add_argument('--pseudo_method', type=str, default='lexical')
parser.add_argument('--dataset', type=str, default='GYAFC_EM')
args = parser.parse_args()
# MODE_MAP = {"yelp":"Yelp", "amazon":"amazon", "gyafc_em":"GYAFC_EM", "gyafc_fr":"GYAFC_FR"}
# args.corpus_mode = MODE_MAP[args.dataset]
global_config.corpus_mode = args.dataset

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def process_lines(text_lines, class_type, prefix_A, prefix_B):
    tmp_lines = [re.sub("\s+", " ", i.replace("\t", " ")).strip()[:200] for i in text_lines if len(i) > 5]
    tmp_lines = ["<s>" + prefix_A + " # " + i + "</s>" if class_type == 1 else "<s>" + prefix_B + " # " + i + "</s>" for i in tmp_lines]
    return tmp_lines


setup_seed(100)

if global_config.corpus_mode == "Yelp":
    """ Reading Yelp data, and add the unsupervised pairs """
    print("Reading Yelp data with pseudo parallel data")
    train_sample, test_sample = [], []
    list_a = open("./pseudo_paired_data_Yelp/" + args.pseudo_method + "_based/merged_0_A_0", "r", encoding="utf-8").readlines()
    list_b = open("./pseudo_paired_data_Yelp/" + args.pseudo_method + "_based/merged_0_B_1", "r", encoding="utf-8").readlines()

    assert len(list_a) == len(list_b)

    for i in range(len(list_a)):
        tmp_a = [j for j in list_a[i].split() if len(j) > 1]
        tmp_b = [j for j in list_b[i].split() if len(j) > 1]

        if len(set(tmp_a) & set(tmp_b)) > 1:
            train_sample.append((process_lines([list_a[i], ], class_type=0, prefix_A="positive", prefix_B="negative")[0],
                                 process_lines([list_b[i], ], class_type=1, prefix_A="positive", prefix_B="negative")[0]))

    test_data_files = {"./pseudo_paired_data_Yelp/reference_0_0": "./pseudo_paired_data_Yelp/reference_0_1",
                       "./pseudo_paired_data_Yelp/reference_1_1": "./pseudo_paired_data_Yelp/reference_1_0"}
    for i in test_data_files.keys():
        test_sample.extend(
            zip(process_lines(open(i, encoding="utf-8").readlines(), class_type=int(i[-1]),
                              prefix_A="positive", prefix_B="negative"),
                process_lines(open(test_data_files[i], encoding="utf-8").readlines(), class_type=int(test_data_files[i][-1]),
                              prefix_A="positive", prefix_B="negative")))

    train_sample = [(i[1], i[0]) for i in train_sample] + train_sample
    # tmp_sample_size = 100
    # train_sample = [(i[1], i[0]) for i in train_sample[:tmp_sample_size]] + train_sample[tmp_sample_size:tmp_sample_size * 2]
    test_sample = [(i[0], i[1]) for i in test_sample]

    """ adding multiple human references """
    human_reference_path = "./Yelp_data/Yelp_comparison/human_references_yelp/"
    multi_candidate_list = []

    file_list_0 = ["reference0.0", "reference1.0", "reference2.0", "reference3.0"]

    fp_list = [open(human_reference_path + str(i), encoding="utf-8").readlines() for i in file_list_0]
    for i in range(len(fp_list[0])):
        multi_candidate_list.append([fp_list[j][i].lower() for j in range(4)])

    file_list_1 = ["reference0.1", "reference1.1", "reference2.1", "reference3.1"]

    fp_list = [open(human_reference_path + str(i), encoding="utf-8").readlines() for i in file_list_1]
    for i in range(len(fp_list[0])):
        multi_candidate_list.append([fp_list[j][i].lower() for j in range(4)])

elif global_config.corpus_mode == "amazon":
    """ Reading amazon data with pseudo parallel data """
    print("Reading amazon data with pseudo parallel data")

    train_sample, test_sample = [], []

    """ reading amazon training sample version 2.0: add post processing """
    list_a = open("./pseudo_paired_data_Amazon/" + args.pseudo_method + "_based/merged_0_A_0", "r", encoding="utf-8").readlines()
    list_b = open("./pseudo_paired_data_Amazon/" + args.pseudo_method + "_based/merged_0_B_1", "r", encoding="utf-8").readlines()

    assert len(list_a) == len(list_b)

    for i in range(len(list_a)):
        tmp_a = [j for j in list_a[i].split() if len(j) > 1]
        tmp_b = [j for j in list_b[i].split() if len(j) > 1]

        if len(set(tmp_a) & set(tmp_b)) > 2 or True:
            train_sample.append((process_lines([list_a[i], ], class_type=0, prefix_A="positive", prefix_B="negative")[0],
                                 process_lines([list_b[i], ], class_type=1, prefix_A="positive", prefix_B="negative")[0]))

    test_data_files = ["amazon_data/reference.0", "amazon_data/reference.1"]

    for i in test_data_files:
        tmp_list = open(i, encoding="utf-8").readlines()
        for one_line in tmp_list:
            if int(i[-1]) == 0:
                test_sample.append((process_lines([one_line.split("\t")[0], ], class_type=0, prefix_A="positive", prefix_B="negative")[0],
                                    process_lines([one_line.split("\t")[1], ], class_type=1, prefix_A="positive", prefix_B="negative")[0],))
            else:
                test_sample.append((process_lines([one_line.split("\t")[0], ], class_type=1, prefix_A="positive", prefix_B="negative")[0],
                                    process_lines([one_line.split("\t")[1], ], class_type=0, prefix_A="positive", prefix_B="negative")[0]))

    train_sample = [(i[1], i[0]) for i in train_sample] + train_sample
    test_sample = [(i[0], i[1]) for i in test_sample]

elif global_config.corpus_mode == "GYAFC_FR":
    """ Reading GYAFC data with formality label """
    print("Reading GYAFC data with formality label")

    train_sample, test_sample = [], []
    train_data_files = {"GYAFC_data/Family_Relationships/train/formal": "GYAFC_data/Family_Relationships/train/informal"}

    for i in train_data_files.keys():
        train_sample.extend(
            zip(process_lines(open(i, encoding="utf-8").readlines(), class_type=1, prefix_A="formal", prefix_B="informal"),
                process_lines(open(train_data_files[i], encoding="utf-8").readlines(), class_type=0, prefix_A="formal", prefix_B="informal")))

    """ Reading test samples from informal -> formal """
    test_data_files = {"GYAFC_data/Family_Relationships/test/formal": "GYAFC_data/Family_Relationships/test/informal.ref0"}

    for i in test_data_files.keys():
        test_sample.extend(
            zip(process_lines(open(i, encoding="utf-8").readlines(), class_type=1, prefix_A="formal", prefix_B="informal"),
                process_lines(open(test_data_files[i], encoding="utf-8").readlines(), class_type=0, prefix_A="formal", prefix_B="informal")))

    multi_candidate_list = []
    fp_list = [open("GYAFC_data/Family_Relationships/test/informal.ref" + str(i), encoding="utf-8").readlines() for i in range(4)]
    for i in range(len(fp_list[0])):
        multi_candidate_list.append([fp_list[j][i] for j in range(4)])

    """ randomly rebuild the labeled pairs """
    train_sample = [(i[0], i[1]) if random.randint(0, 100) > 50 else (i[1], i[0]) for i in train_sample]

else:
    """ Reading GYAFC data with formality label """
    print("Reading GYAFC data with formality label")

    train_sample, test_sample = [], []
    train_data_files = {"GYAFC_data/Entertainment_Music/train/formal": "GYAFC_data/Entertainment_Music/train/informal"}

    for i in train_data_files.keys():
        train_sample.extend(
            zip(process_lines(open(i, encoding="utf-8").readlines(), class_type=1, prefix_A="formal",
                              prefix_B="informal"),
                process_lines(open(train_data_files[i], encoding="utf-8").readlines(), class_type=0,
                              prefix_A="formal", prefix_B="informal")))

    """ Reading test samples from formal -> informal """
    test_data_files = {"GYAFC_data/Entertainment_Music/test/formal": "GYAFC_data/Entertainment_Music/test/informal.ref0"}

    for i in test_data_files.keys():
        test_sample.extend(
            zip(process_lines(open(i, encoding="utf-8").readlines(), class_type=1, prefix_A="formal",
                              prefix_B="informal"),
                process_lines(open(test_data_files[i], encoding="utf-8").readlines(), class_type=0, prefix_A="formal",
                              prefix_B="informal")))

    multi_candidate_list = []
    fp_list = [open("GYAFC_data/Entertainment_Music/test/informal.ref" + str(i), encoding="utf-8").readlines() for i in
               range(4)]
    for i in range(len(fp_list[0])):
        multi_candidate_list.append([fp_list[j][i] for j in range(4)])

    """ randomly rebuild the labeled pairs """
    train_sample = [(i[0], i[1]) if random.randint(0, 100) > 50 else (i[1], i[0]) for i in train_sample]

""" Using all samples """
data_train = train_sample
data_test_collection = [test_sample, ]

print('Train Dataset sizes: %d' % (len(data_train)))
print('Test Dataset sizes: %d' % (sum([len(one_data) for one_data in data_test_collection])))


def train_process(model, train_data):
    train_epoch = global_config.start_from_epoch

    best_score = {"epoch": 0, "all_loss": 0}
    running_log_name = None

    while train_epoch < global_config.num_epochs:
        print("\n********* Epoch {} ***********".format(train_epoch))
        summary_steps = 0

        random.seed(100 + train_epoch)
        random.shuffle(train_data)

        train_batches = get_batches(train_data, global_config.batch_size)

        if global_config.pure_unsupervised_training or train_epoch >= global_config.supervised_epoch_num:
            model.RL_training = True
            model.supervised_loss_decay = float(args.decay_rate) ** (train_epoch + 1)


        for batch in train_batches:
            supervised_loss, cyclic_loss, GAN_dis_loss, GAN_gen_loss, transferred_sen_text, _ = model.batch_train(batch, train_epoch, args)
            summary_steps += 1
            if summary_steps % global_config.batch_loss_print_interval == 0:
                print(train_epoch, summary_steps, "supervised loss", supervised_loss, "cyclic loss", cyclic_loss, "GAN_dis loss", GAN_dis_loss, "GAN_gen loss", GAN_gen_loss)
                # print(transferred_sen_text)

        best_score, current_eval_score, style_accuracy = evaluate_process(model, data_test_collection, train_epoch, best_score)

        if running_log_name is None:
            running_log_path = "./running_log/" + args.dataset + '/'
            if not os.path.exists(running_log_path):
                os.mkdir(running_log_path)
            running_log_name = running_log_path + "decay_rate:" + str(args.decay_rate) + \
                               "_cyclic:" + str(args.cyclic) + "_style:" + str(args.style) + ".txt"
            open(running_log_name, "w").write("Corpus mode: " + global_config.corpus_mode + "\n" + "Pair mode: " + args.pseudo_method + "\n" +
                                              "decay_rate: " + str(args.decay_rate) + "\n" + "cyclic: " + str(args.cyclic) + "\n" +
                                              "style: " + str(args.style) + "\n")

        if running_log_name:
            with open(running_log_name, "a") as fp:
                fp.write("\nEpoch: " + str(train_epoch) + " supervised loss decay: " + str(model.supervised_loss_decay))
                fp.write("\nEpoch: " + str(train_epoch) + " BLEU Score: " + str(current_eval_score) + " Style Accuracy: " + str(style_accuracy) + "\n")

        train_epoch += 1

    with open("xyf_sh/tmp_best_result", "a") as fp:
        save_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        fp.write(save_time + " Best score result: " + str(best_score) + "\n")


def evaluate_process(model, data_test_collection, train_epoch, best_score=None, preview=False, fast_infer=False):
    all_eval_score = 0
    style_accuracy = None

    for tmp_i in range(len(data_test_collection)):
        test_batches = get_batches(data_test_collection[tmp_i], global_config.batch_size)

        all_transferred_sentences, all_gold_sentences = [], []
        all_test_loss = []
        all_style_cls_pred, all_style_cls_gold = [], []
        for batch in tqdm(test_batches):
            if fast_infer is False:
                supervised_loss, cyclic_loss, GAN_dis_loss, GAN_gen_loss, transferred_sen_text, style_CLS_res = model.batch_eval(batch)
                all_test_loss.append(supervised_loss)
            else:
                transferred_sen_text, style_CLS_res = model.batch_infer(batch)
                all_test_loss.append(-1)

            all_transferred_sentences.extend(transferred_sen_text)
            all_gold_sentences.extend([i[1] for i in batch])
            all_style_cls_pred.extend(style_CLS_res[0])
            all_style_cls_gold.extend(style_CLS_res[1])

        if global_config.print_all_predictions:
            saved_models_path = "./saved_models/" + args.dataset + '/'
            if not os.path.exists(saved_models_path):
                os.mkdir(saved_models_path)
            with open(saved_models_path +"decay_rate:" + str(args.decay_rate) + "_cyclic:" + str(args.cyclic) + \
                      "_style:" + str(args.style) + "_epoch:" + str(train_epoch) + ".txt", "w", encoding="utf-8") as fp:
                for i in all_transferred_sentences:
                    fp.write((i[i.index("#") + 1:]).strip() + "\n")

        all_transferred_sentences = [word_tokenize(i[i.index("#") + 1:]) if "#" in i else word_tokenize(i) for i in all_transferred_sentences]
        if global_config.corpus_mode == "Yelp":
            # """ Using one reference """
            # all_gold_sentences = [[word_tokenize(v.replace("</s>", "").split("tive # ")[-1]), ] for k, v in enumerate(all_gold_sentences)]
            """ Using multiple reference """
            assert len(multi_candidate_list) == len(all_transferred_sentences)
            all_gold_sentences = [[word_tokenize(multi_candidate_list[k][0]), word_tokenize(multi_candidate_list[k][1]),
                                   word_tokenize(multi_candidate_list[k][2]), word_tokenize(multi_candidate_list[k][3])] for k, v in enumerate(multi_candidate_list)]

        elif global_config.corpus_mode == "amazon":
            """ Using one reference """
            all_gold_sentences = [[word_tokenize(v.replace("</s>", "").split("tive # ")[-1]), ] for k, v in enumerate(all_gold_sentences)]

        else:
            # """ Using one reference """
            # all_gold_sentences = [[word_tokenize(v.replace("</s>", "").split("formal # ")[-1]), ] for k, v in enumerate(all_gold_sentences)]
            assert len(multi_candidate_list) == len(all_transferred_sentences)
            all_gold_sentences = [[word_tokenize(multi_candidate_list[k][0]), word_tokenize(multi_candidate_list[k][1]),
                                   word_tokenize(multi_candidate_list[k][2]), word_tokenize(multi_candidate_list[k][3])] for k, v in enumerate(multi_candidate_list)]

        print("\nTest Result in epoch:", train_epoch)
        bleu_score = corpus_bleu(list_of_references=all_gold_sentences, hypotheses=all_transferred_sentences)
        all_eval_score += bleu_score

        # [print(data_test_collection[tmp_i][-i], "\n", " ".join(all_transferred_sentences[-i])) for i in range(20)]
        # [print(all_gold_sentences[i], "\n", all_transferred_sentences[i], "\n\n") for i in range(10)]

        if any(all_test_loss):
            print("Test loss:", np.mean(all_test_loss))
        print("BLEU Score:", bleu_score)

        if len(all_style_cls_pred) > 1:
            style_accuracy = accuracy_score(y_pred=all_style_cls_pred, y_true=all_style_cls_gold)
            print("Style accuracy: ", style_accuracy, "\n")

    current_eval_score = all_eval_score / len(data_test_collection)

    if best_score:
        if current_eval_score > best_score["all_loss"]:
            best_score = {"epoch": train_epoch, "all_loss": current_eval_score}

    if global_config.save_model and global_config.train and train_epoch >= 1:
        model.save_model("./saved_models/best_model_" + str(running_random_number) + "_epoch" + str(train_epoch) + "_" + str(current_eval_score)[:7] + ".pth")

    return best_score, current_eval_score, style_accuracy


if __name__ == '__main__':

    model = Model(args.dataset)

    train_mode = global_config.train

    if train_mode:
        if global_config.start_from_epoch != 0:
            load_model_path = global_config.load_model_path
            model.load_model(load_model_path)

        train_process(model, data_train)
    else:
        load_model_path = global_config.load_model_path
        model.load_model(load_model_path)
        evaluate_process(model, data_test_collection, -999, fast_infer=True)
