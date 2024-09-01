#! /usr/bin/python3
import time
import random
import subprocess
from manager import GPUManager
import argparse
# dataset_list = ["GYAFC_FR_INFORMAL", "GYAFC_FR_FORMAL", "GYAFC_EM_INFORMAL", "GYAFC_EM_FORMAL", \
#            "SHAKESPEARE-MODERN", "SHAKESPEARE-ORIGINAL", "PTB-FUTURE", "PTB-REMOVAL"]
# dataset_list = ["PTB-FUTURE", "PTB-REMOVAL"]
dataset_list = ["GYAFC_FR_FORMAL", "SHAKESPEARE-MODERN"]
# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, default="gyafc_em")
# args = parser.parse_args()

def get_mission_queue(cmd_command):
    mission_queue = []
    peft_type_list = ["LORA"]
    lr_list = [2e-4, 5e-4, 1e-3]
    for peft_type in peft_type_list:
        for ds in dataset_list:
            for lr in lr_list:
                cmd_params = f"--base_model /home/xyf/LanguageModel/huggingface/yahma/llama-7b-hf " \
                             f"--data_path /home/xyf/PycharmProjects/BENCHMARKforTST/alpaca-lora-main/alpaca_data_cleaned.json " \
                             f"--output_dir ./output/ " \
                             f"--num_epochs 5 " \
                             f"--learning_rate {lr} " \
                             f"--cutoff_len 512 " \
                             f"--adapter_name {peft_type} " \
                             f"--num_virtual_tokens 64 " \
                             f"--wandb_project LLAMA2 " \
                             f"--batch_size 256 " \
                             f"--micro_batch_size 32 " \
                             f"--lora_r 16 " \
                             f"--dataset {ds} " \

                cmd = f"{cmd_command} {cmd_params}"
                mission_queue.append(cmd)

    return mission_queue

gm = GPUManager()
cmd_command = 'python TST_finetune.py '
mission_queue = get_mission_queue(cmd_command)
total = len(mission_queue)
finished = 0
running = 0
p = []
min_gpu_number = 1  # 最小GPU数量，多于这个数值才会开始执行训练任务。
time_interval = 120  # 监控GPU状态的频率，单位秒。

while finished + running < total:
    localtime = time.asctime(time.localtime(time.time()))
    gpu_av = gm.choose_no_task_gpu()
    # 在每轮epoch当中仅提交1个GPU计算任务
    if len(gpu_av) >= min_gpu_number:
        # 为了保证服务器上所有GPU负载均衡，从所有空闲GPU当中随机选择一个执行本轮次的计算任务
        gpu_index = random.sample(gpu_av, min_gpu_number)[:min_gpu_number]
        gpu_index_str = ','.join(map(str, gpu_index))

        # 确保能够连接到wanbd
        # response = requests.get("https://www.wandb.ai")
        # assert response.status_code == 200
        cmd_ = 'CUDA_VISIBLE_DEVICES=' + gpu_index_str + ' ' + mission_queue.pop(0)  # mission_queue当中的任务采用先进先出优先级策略
        print(f'Mission : {cmd_}\nRUN ON GPU : {gpu_index_str}\nStarted @ {localtime}\n')
        # subprocess.call(cmd_, shell=True)
        p.append(subprocess.Popen(cmd_, shell=True))
        running += 1
        time.sleep(time_interval)  # 等待NVIDIA CUDA代码库初始化并启动


    else:  # 如果服务器上所有GPU都已经满载则不提交GPU计算任务
        pass
        # print(f'Keep Looking @ {localtime} \r')

    new_p = []  # 用来存储已经提交到GPU但是还没结束计算的进程
    for i in range(len(p)):
        if p[i].poll() != None:
            running -= 1
            finished += 1
        else:
            new_p.append(p[i])
    # if len(new_p) == len(p):  # 此时说明已提交GPU的进程队列当中没有进程被执行完
    #     time.sleep(time_interval)
    #     1
    p = new_p

for i in range(len(p)):  # mission_queue队列当中的所有GPU计算任务均已提交，等待GPU计算完毕结束主进程
    p[i].wait()

print('Mission Complete ! Checking GPU Process Over ! ')
