import pandas as pd
import os
import datasets
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import csv
import argparse

def load_dataset(dataset_name):
    train_data = []
    test_data = []
    with open(os.path.join("../data", dataset_name, "train.0"), "r") as file:
        for line in file.readlines():
            train_data.append({'text': line.strip(),
                         'label': 0
            })
    with open(os.path.join("../data", dataset_name, "train.1"), "r") as file:
        for line in file.readlines():
            train_data.append({'text': line.strip(),
                         'label': 1
            })

    with open(os.path.join("../data", dataset_name, "test.0"), "r") as file:
        for line in file.readlines():
            test_data.append({'text': line.strip(),
                         'label': 0
            })

    with open(os.path.join("../data", dataset_name, "test.1"), "r") as file:
        for line in file.readlines():
            test_data.append({'text': line.strip(),
                         'label': 1
            })
    return train_data, test_data

def load_csv_file(dataset_name):
    with open(os.path.join("../data", dataset_name, "train.0"), "r") as f1, open(os.path.join("../data", dataset_name, "train.1"), "r") as f2, \
            open(f'./classifier/{dataset_name}/train.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['label', 'text'])
        for line in f1.readlines():
            if len(line.strip().split(' '))>64:
                line = ' '.join(line.strip().split(' ')[:64])
            writer.writerow([0,f'{line.strip()}'])
        for line in f2.readlines():
            writer.writerow([1,f'{line.strip()}'])

    with open(os.path.join("../data", dataset_name, "test.0"), "r") as f1, \
            open(f'./classifier/{dataset_name}/test.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['label', 'text'])
        for line in f1.readlines():
            writer.writerow([0,f'{line.strip()}'])
        if 'style' not in dataset_name:
            with open(os.path.join("../data", dataset_name, "test.1"), "r") as f2:
                for line in f2.readlines():
                    writer.writerow([1,f'{line.strip()}'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='amazon')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    output_dir = f'./classifier/{dataset_name}'
    load_csv_file(dataset_name)

    data_files = {"train": f"./classifier/{dataset_name}/train.csv",
                  "test": f"./classifier/{dataset_name}/test.csv"}
    datasets = datasets.load_dataset("csv", data_files=data_files)

    # load model and tokenizer and define length of the text sequence
    model = RobertaForSequenceClassification.from_pretrained('/home/xyf/LanguageModel/roberta-base')
    tokenizer = RobertaTokenizer.from_pretrained('/home/xyf/LanguageModel/roberta-base', max_length=64)


    # define a function that will tokenize the model, and will return the relevant inputs for the model
    def tokenization(batched_text):
        return tokenizer(batched_text['text'], padding=True, truncation=True)


    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")


    train_data = datasets['train'].map(tokenization, batched=True, batch_size=len(datasets['train']))
    test_data = datasets['test'].map(tokenization, batched=True, batch_size=len(datasets['test']))

    train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    test_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])


    # define accuracy metrics
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }


    # define the training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        save_total_limit=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        disable_tqdm=False,
        greater_is_better=True,
        metric_for_best_model="eval_accuracy",
        load_best_model_at_end=True,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=20,
        fp16=True,
        dataloader_num_workers=8,
        report_to="none"
    )
    # instantiate the trainer class and check for available devices
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=test_data
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # train the model
    trainer.train()
    model.save_pretrained(output_dir)
    trainer.evaluate()