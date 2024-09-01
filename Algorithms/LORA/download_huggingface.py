from huggingface_hub import snapshot_download
import os
import time
# export HF_ENDPOINT=https://hf-mirror.com

model_name_list = ['yahma/llama-7b-hf']

for model_name in model_name_list:
    tick = time.time()
    while True:
        try:
            snapshot_download(repo_id=model_name,
                              resume_download=True,
                              ignore_patterns=["*.h5", "*.ot", "*.msgpack", "*.onnx", "*.safetensors"],
                              local_dir=f"/home/xyf/LanguageModel/huggingface/{model_name}")
            break
        except Exception as e:
            print(e)

    print(model_name, '下载完成')
    print('time %.2f' % (time.time() - tick))