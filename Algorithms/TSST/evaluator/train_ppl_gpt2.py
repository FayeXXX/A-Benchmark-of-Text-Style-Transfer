import os
import torch
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from tqdm import tqdm
from config import LABEL_MAP
from datasets import load_dataset


gpt_config = GPT2Config.from_pretrained("/home/xyf/LanguageModel/gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("/home/xyf/LanguageModel/gpt2")
model = GPT2LMHeadModel.from_pretrained("/home/xyf/LanguageModel/gpt2", config=gpt_config)





def tokenize_function(examples):
    return tokenizer(examples["text"], return_special_tokens_mask=True)


def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


def compute_ppl_GPT(self, sentences):
    encodings = self.gpt2_tokenizer("\n\n".join(sentences), return_tensors="pt")
    max_length = self.gpt2.config.n_positions
    stride = 512
    nlls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = self.gpt2(input_ids, labels=target_ids)
            neg_log_likelihood = outputs[0] * trg_len

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="yelp")
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--save_model', type=str, default='./evaluator/lm/')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()


    block_size = tokenizer.model_max_length
    datasets = load_dataset("text", data_files={"train": 'train.txt', "validation": 'dev.txt'})

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="output",
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
        # Add other desired training arguments
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset
    )

    trainer.train()
    print('training done')
    model.save_pretrained("fine-tuned")

