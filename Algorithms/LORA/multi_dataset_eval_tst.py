from concurrent.futures import ProcessPoolExecutor
import queue
import subprocess


def evaluate(dataset, gpu):
    print('*******dataset:', dataset)
    command = f"CUDA_VISIBLE_DEVICES={gpu} python evaluate_tst2_all.py \
               --model LLaMA-7B \
               --adapter LORA \
               --dataset {dataset} \
               --base_model '/home/xyf/LanguageModel/huggingface/yahma/llama-7b-hf' \
               --lora_weights '/home/xyf/PycharmProjects/BENCHMARKforTST/alpaca-lora-main/r=16-q_proj-k_proj-v_proj-o_proj-up_proj-down_proj'"

    result = subprocess.run(command, shell=True, text=True, capture_output=False)
    print(f"Evaluation results for dataset {dataset} on GPU {gpu}:\n{result.stdout}")
    return gpu

datasets = ['GYAFC-FORMAL', 'GYAFC-INFORMAL', 'YELP-NEGATIVE', 'YELP-POSITIVE',
            'GYAFC_EM_FORMAL', 'GYAFC_EM_INFORMAL', 'AMAZON-NEGATIVE', 'AMAZON-POSITIVE',
            'SHAKESPEARE-MODERN', 'SHAKESPEARE-ORIGINAL', 'PTB-FUTURE', 'PTB-REMOVAL']
gpus = [3]
tasks_queue = queue.Queue()
gpu_queue = queue.Queue()

for gpu in gpus:
    gpu_queue.put(gpu)
for task in datasets:
    tasks_queue.put(task)

num_processes = min(len(datasets), len(gpus))  # number of processes to run in parallel

with ProcessPoolExecutor(max_workers=num_processes) as executor:
    futures = [executor.submit(evaluate, tasks_queue.get(), gpu_queue.get()) for i in range(num_processes)]
    for future in futures:
        gpu_id = future.result()
        gpu_queue.put(gpu_id)
        if tasks_queue.qsize() > 0:
            futures.append(executor.submit(evaluate, tasks_queue.get(), gpu_queue.get()))

