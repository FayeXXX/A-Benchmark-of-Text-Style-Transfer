import json
import os
import pathlib
import json

dir = '/datasets/styleptb_json/TFU'
json_file_path = os.path.join(dir, 'test.jsonl')
new_dir = '/datasets/styleptb/TFU'
pathlib.Path(new_dir).mkdir(parents=True, exist_ok=True)

source_file_path = os.path.join(new_dir,'test.0')
target_file_path = os.path.join(new_dir,'test.1')

with open(source_file_path, 'w') as source_file, open(target_file_path, 'w') as target_file:
    with open(json_file_path, 'r') as json_file:
        for line in json_file:

            data = json.loads(line.strip())

            source_text = data['src']  
            target_text = data['trg'] #

            source_file.write(source_text + '\n')
            target_file.write(target_text + '\n')

print("save to '{}', '{}'".format(source_file_path, target_file_path))
