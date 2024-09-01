import os
import math
import torch
import argparse
from nltk.translate.bleu_score import corpus_bleu
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, GPT2LMHeadModel, GPT2TokenizerFast
from bleurt import score
from comet.models import load_from_checkpoint
from train_textcnn_cla import TextCNN
from utils.dataset import collate_fn,LABEL_MAP
from tqdm import tqdm
import sys
sys.path.append(".")
sys.path.append("..")
from utils.dataset import read_file
from statistics import mean

def prepare_data(args):
    labels = LABEL_MAP[args.dataset]
#prepare the training data
    for mode in ["train", "dev", "test"]:
        with open('{}.txt'.format(mode),'w') as f3:
            for label in labels:
                with open(os.path.join(args.data_dir, "{}.{}".format(mode, label)),'r') as f1:
                    f3.writelines(f1.readlines())


class Evaluator:
    def __init__(self, dataset_name="yelp", device="cpu"):
        
        self.cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(os.path.join(os.path.dirname(self.cur_dir), "data"), dataset_name)

        self.dataset_name = dataset_name
        
        # original data and references for self-bleu and ref-bleu
        self.ori_data = self.__get_data__(file_name="test")
        self.ref_data = self.__get_data__(file_name="reference")

        # classifier for acc
        self.device = device

        classifier_path_bert = os.path.join(os.path.join(self.cur_dir, "classifier"), "{}/bert".format(dataset_name))
        self.bert = BertForSequenceClassification.from_pretrained(classifier_path_bert).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(classifier_path_bert)

        classifier_path_cnn = os.path.join(os.path.join(self.cur_dir, "classifier"), "{}/textcnn/textcnn.chkpt".format(dataset_name))
        filter_sizes = [1, 2, 3, 4, 5]
        num_filters = [128, 128, 128, 128, 128]
        cnn_embed_dim = 300
        cnn_dropout = 0.5
        self.cnn = TextCNN(cnn_embed_dim, len(self.tokenizer), filter_sizes,
                           num_filters, None, dropout=cnn_dropout)
        self.cnn.to(device).eval()
        self.cnn.load_state_dict(torch.load(classifier_path_cnn))

        # language model for ppl
        self.ppl_lm_pathGPT = os.path.join(os.path.join(self.cur_dir, "lm"), "{}/gpt".format(dataset_name))
        self.ppl_tokenizerGPT = GPT2TokenizerFast.from_pretrained(self.ppl_lm_pathGPT)
        self.ppl_modelGPT = GPT2LMHeadModel.from_pretrained(self.ppl_lm_pathGPT).to(self.device)

        # bleurt model
        self.BLEURT = score.BleurtScorer('/home/xyf/LanguageModel/bleurt-large-512')

        #comet model
        # self.COMET = load_from_checkpoint('/home/xyf/LanguageModel/wmt-large-da-estimator-1719')

    def __get_data__(self, file_name="test"):
        data = []
        for label in LABEL_MAP[self.dataset_name]:
            if self.dataset_name == "gyafc" and file_name == "reference":
                gyafc_ref = [read_file(os.path.join(self.data_dir, "{}.{}.{}".format(file_name, label, i))) for i in range(4)]
                label_data = list(zip(gyafc_ref[0], gyafc_ref[1], gyafc_ref[2], gyafc_ref[3]))
                data += [[sent.split() for sent in sents] for sents in label_data]
            else:
                label_data = read_file(os.path.join(self.data_dir, "{}.{}".format(file_name, label)))
                data += [[sent.split()] for sent in label_data]
        return data

    def get_ref_bleu(self, seg_sents):
        try:
            assert len(seg_sents) == len(self.ref_data)
        except:
            print(len(seg_sents))
        return corpus_bleu(self.ref_data, seg_sents)

    def get_self_bleu(self, seg_sents):
        try:
            assert len(seg_sents) == len(self.ori_data)
        except:
            print(len(seg_sents))
        return corpus_bleu(self.ori_data, seg_sents)


    def compute_ppl_GPT(self, sentences):
        encodings = self.ppl_tokenizerGPT("\n\n".join(sentences), return_tensors="pt")
        max_length = self.ppl_modelGPT.config.n_positions
        stride = 512
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.ppl_modelGPT(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl.item()


    def get_acc(self, sents, labels):
        batch = 32
        right_count_bert = 0
        total_count_bert = 0

        right_count_cnn = 0
        total_count_cnn = 0

        for i in range(0, len(sents), batch):
            predict_bert = self.bert(**(self.tokenizer(sents[i:i+batch], max_length=256, padding=True, return_tensors="pt").to(self.device)))[0]
            preds_bert = torch.argmax(predict_bert, dim=-1).cpu()
            l = torch.LongTensor(labels[i:i+batch])
            right_count_bert += preds_bert.ne(l).sum()
            total_count_bert += preds_bert.size(0)

            seq = collate_fn([self.tokenizer.encode(sent) for sent in sents[i:i+batch]], pad_token_id=1).to(self.device)
            predict_cnn = self.cnn(seq)
            preds_cnn = torch.argmax(predict_cnn,dim=-1).cpu()
            right_count_cnn += preds_cnn.ne(l).sum()
            total_count_cnn += preds_cnn.size(0)

        assert len(sents) == total_count_bert
        assert len(sents) == total_count_cnn
        return right_count_bert.item()/total_count_bert * 100, right_count_cnn.item()/total_count_cnn * 100


    def get_bleurt(self, transfered_sents):
        src = self.ori_data
        src = [' '.join(item[0]) for item in src]
        out = transfered_sents
        ref = self.ref_data
        if self.dataset_name == "gyafc":
            pass
        else:
            ref = [' '.join(item[0]) for item in ref]
        sample = []

        # BLEURT score
        sample.append(round(mean(self.BLEURT.score(references=src, candidates=out)), 4))
        sample.append(round(mean(self.BLEURT.score(references=ref, candidates=out)), 4))
        return sample

    def evaluate(self, transfered_sents, labels):
        #acc
        acc_bert, acc_cnn = self.get_acc(transfered_sents, labels)

        #ppl
        ppl_gpt = self.compute_ppl_GPT(transfered_sents)

        #content
        seg_sents = [sent.split() for sent in transfered_sents]
        bleurt_score = self.get_bleurt(transfered_sents)
        self_bleu = self.get_self_bleu(seg_sents)
        ref_bleu = self.get_ref_bleu(seg_sents)

        hm = 2.0/(1.0/acc+1/.0/self_bleu)
        gm = math.pow(acc_bert * self_bleu, 1.0/2.0)
    
        # eval_str = "ACC_bert: {:.1f} \tACC_cnn: {:.1f} \tself-BLEU: {:.2f} \tref-BLEU: {:.2f} \tPPL: {:.0f} \tGM: {:.2f}".format(acc_bert, acc_cnn, self_bleu, ref_bleu, ppl, gm)
        return eval_str, (acc, self_bleu, ref_bleu, ppl, gm)

    def evaluate_file(self, result_file):
        transfered_sents = []
        labels = []
        for i, label in enumerate(LABEL_MAP[self.dataset_name]):
            sents = read_file("{}.{}".format(result_file, label))
            transfered_sents += sents
            labels += [i] * len(sents)
        
        eval_str, (acc, self_bleu, ref_bleu, ppl, gm) = self.evaluate(transfered_sents, labels)
        return eval_str, (acc, self_bleu, ref_bleu, ppl, gm)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="yelp")
    parser.add_argument('--datadir', type=str, default="outputs")
    parser.add_argument('--file', type=str, default="all",help='../outputs/yelp/epoch_7')
    parser.add_argument('--cpu', action='store', default=False)
    args = parser.parse_args()

    dataset_name = args.dataset
    print(f"Evaluating dataset: {dataset_name}")

    if args.file == "all":
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        outputs_dir = os.path.join(os.path.join(os.path.dirname(cur_dir), args.datadir), dataset_name)
        result_files = set()
        for file in os.listdir(outputs_dir):
            file_split = file.split(".")
            if len(file_split) == 2 and file_split[0] != "log":
                result_files.add(os.path.join(outputs_dir, file_split[0]))
        result_files = sorted(list(result_files))
    else:
        result_files = [args.file]


    device = "cuda" if not args.cpu and torch.cuda.is_available() else "cpu"
    evaluator = Evaluator(dataset_name, device=device)
    for result_file in result_files:
        print(result_file, end="\t")
        eval_str, (acc, self_bleu, ref_bleu, ppl, gm) = evaluator.evaluate_file(result_file)
        print(eval_str)