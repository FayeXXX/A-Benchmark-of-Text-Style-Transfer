import sys
import os
import fire
import copy
import json
import torch
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.insert(0, "/home/xyf/PycharmProjects/BENCHMARKforTST/alpaca-lora-main")
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

DATASET_TO_INSTRUCTION_MAPPING = {
    "YELP-NEGATIVE": "Please change the sentiment of the following sentence to be more positive.",
    "YELP-POSITIVE": "Please change the sentiment of the following sentence to be more negative.",
    "GYAFC-INFORMAL": "Please rewrite the following sentence to be more formal.",
    "GYAFC-FORMAL": "Please rewrite the following sentence to be more informal.",
}


# 获取路径的最后三级
def get_last_three_levels(path):
    parts = path.split(os.path.sep)[-3:]
    return os.path.join(*parts)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['GYAFC-INFORMAL', 'GYAFC-FORMAL', 'YELP-NEGATIVE', 'YELP-POSITIVE'],
                        required=True)
    parser.add_argument('--model', choices=['LLaMA-7B', 'BLOOM-7B', 'GPT-j-6B'], required=True)
    parser.add_argument('--adapter',
                        choices=['NONE', 'FT',
                                 'LORA', 'SALORA', 'ADALORA',
                                 'PROMPT_TUNING', 'PREFIX_TUNING',
                                 'AdapterP', 'AdapterH', 'Parallel', 'Scaled_Parallel'],
                        required=True)
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--lora_weights', required=True)
    parser.add_argument('--load_8bit', action='store_true', default=False)

    return parser.parse_args()


def main(
        load_8bit: bool = False,
        base_model: str = "",
        lora_weights: str = "tloen/alpaca-lora-7b",
        share_gradio: bool = False,
):
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    args = parse_args()

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if args.adapter in ['LORA', 'SALORA', 'ADALORA', 'PROMPT_TUNING', 'PREFIX_TUNING']:
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        ).to(device)
    else:
        model.to(device)

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
            instruction,
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=128,
            **kwargs,
    ):
        prompt = generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s, skip_special_tokens=True)
        return output.split("### Response:")[1].strip()

    dataset = load_data(args)
    total = len(dataset)
    if args.adapter in ['LORA', 'SALORA', 'ADALORA', 'PROMPT_TUNING', 'PREFIX_TUNING']:
        exp_dir = os.path.join("./experiment", get_last_three_levels(lora_weights))
    elif args.adapter == "FT":
        exp_dir = os.path.join("./experiment", "llama-7b-hf/FT")
    else:
        exp_dir = os.path.join("./experiment", "llama-7b-hf/NONE")

    os.makedirs(exp_dir, exist_ok=True)
    save_file = os.path.join(exp_dir, f"{args.dataset}.json")

    output_data = []

    pbar = tqdm(total=total)
    for idx, data in enumerate(dataset):
        instruction = data.get('instruction')
        input = data.get('input')
        response = evaluate(instruction, input)
        new_data = copy.deepcopy(data)
        new_data['output'] = response
        output_data.append(new_data)
        print(' ')
        print('---------------')
        print(input)
        print('\n')
        print(response)
        print('---------------')
        with open(save_file, 'w+') as f:
            json.dump(output_data, f, indent=4)
        pbar.update(1)
    pbar.close()

    print('\n')
    print('test finished')


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

### Instruction:
{instruction}

### Response:
"""


def load_data(args) -> list:
    """
    read data from dataset file
    Args:
        args:

    Returns:

    """
    input_file_path = f'data/{args.dataset}/test.txt'

    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"can not find dataset file : {input_file_path}")
    if args.dataset in DATASET_TO_INSTRUCTION_MAPPING.keys():
        instruction = DATASET_TO_INSTRUCTION_MAPPING[args.dataset]
    else:
        raise FileNotFoundError(f"can not find dataset instruction")

    sentences = []
    if args.dataset.startswith("YELP"):
        ref_file_path = f'data/{args.dataset}/ref.txt'
        with open(input_file_path, "r") as file1, open(ref_file_path, "r") as file2:
            inputs = file1.readlines()
            refs = file2.readlines()
            for input, ref in zip(inputs, refs):
                sentences.append({"instruction": instruction,
                                  "input": input.strip().lower(),
                                  "ref": [ref.strip().lower()]}
                )
    else:
        directory = f'data/{args.dataset}'
        ref_file_paths = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.startswith("ref")]

        with open(input_file_path, "r") as file0, open(ref_file_paths[0], "r") as file1, open(ref_file_paths[1], "r") as file2, \
                open(ref_file_paths[2], "r") as file3, open(ref_file_paths[3], "r") as file4:

            # 逐行读取每个文件的内容
            lines0 = file0.readlines()
            lines1 = file1.readlines()
            lines2 = file2.readlines()
            lines3 = file3.readlines()
            lines4 = file4.readlines()

            assert len(lines0) == len(lines1) == len(lines2) == len(lines3) == len(lines4)

            for i in range(len(lines0)):
                sentences.append({"instruction": instruction,
                                  "input": lines0[i].strip().lower(),
                                  "ref": [lines1[i].strip().lower(), lines2[i].strip().lower(),
                                          lines3[i].strip().lower(), lines4[i].strip().lower()]
                                  }
                                 )
    return sentences


if __name__ == "__main__":
    fire.Fire(main)
