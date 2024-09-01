from transformers import RobertaForSequenceClassification
from Model import StyleClassifier
from sklearn.metrics import accuracy_score
import os

# print(bool(30%3))


# running_log_path = "./running_log/" + 'yelp' + '/'
# if not os.path.exists(running_log_path):
#     os.mkdir(running_log_path)


# with open('./saved_models/GYAFC_FR/decay_rate:0.95_cyclic:0.8_style:1.2_epoch:6.txt','r') as f:
with open('./amazon_data/sentiment.test.0','r') as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]
style_CLS = StyleClassifier('amazon')
_, style_CLS_pred = style_CLS.binary_cls(lines)
style_CLS_pred = style_CLS_pred.detach().cpu().numpy().tolist()
all_style_cls_gold = [1]*len(lines)

style_accuracy = accuracy_score(y_pred=style_CLS_pred, y_true=all_style_cls_gold)
print("Style accuracy: ", style_accuracy, "\n")