import os, shutil

def rename():
    file_name = []
    MAP = {}
    for file in os.listdir('/home/xyf/PycharmProjects/BENCHMARKforTST/mix_results/bsrr/yelp/running_log'):
        with open(f'/home/xyf/PycharmProjects/BENCHMARKforTST/mix_results/bsrr/yelp/running_log/{file}', 'r') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        tp = lines[1].split(':')[1].strip()
        if tp == 'semantic':
            file_name.append(file.split('.')[0][-4:])
            cyclic = lines[3].split(' ')[1]
            style = lines[4].split(' ')[1]
            decay_rate = lines[2].split(' ')[1]
            new_name = "decay_rate:" + decay_rate + "_cyclic:" + cyclic + "_style:" + style
            MAP[file.split('.')[0][-4:]] = new_name

    dirs = '/home/xyf/PycharmProjects/BENCHMARKforTST/generated_all_algs/yelp/1eval_outs/bsrr'
    count = 0
    for file in os.listdir(dirs):
        if os.path.isdir(dirs + '/' + file):
            name = file.split('_')[0]
            if name in file_name:
                shutil.copytree(f'{dirs}/{file}', f'{dirs}_new/{MAP[name]}_{file.split("_")[1]}')
                count += 1
    return


rename()

# for file in os.listdir('/home/xyf/PycharmProjects/BENCHMARKforTST/mix_results/bsrr/yelp/running_log'):
#     if file.split('.')[0][-4:]=='9380':
#         with open(f'/home/xyf/PycharmProjects/BENCHMARKforTST/mix_results/bsrr/yelp/running_log/{file}','r') as f:
#             lines = f.readlines()
#         lines = [line.strip() for line in lines]
#         cyclic = lines[3].split(' ')[1]
#         style = lines[4].split(' ')[1]
#         print(f'decay_rate:{lines[2]},cyclic:{cyclic},style:{style}')



# for file in os.listdir('/home/xyf/PycharmProjects/BENCHMARKforTST/mix_results/bsrr/yelp/running_log'):
#     with open(f'/home/xyf/PycharmProjects/BENCHMARKforTST/mix_results/bsrr/yelp/running_log/{file}','r') as f:
#         lines = f.readlines()
#     lines = [line.strip() for line in lines]
#     if lines[1]=='Pair mode: lexical' and lines[4]=='style: 0.8' and lines[3]=='cyclic: 1.0':
#         if lines[2]=='decay_rate: 0.6':
#             print(f'lines[2]==0.6 and lines[3]==1.0 and lines[4]==0.8:{file}')
#         elif lines[2]=='decay_rate: 0.7':
#             print(f'lines[2]==0.7 and lines[3]==1.0 and lines[4]==0.8:{file}')
#         elif lines[2]=='decay_rate: 0.8':
#             print(f'lines[2]==0.8 and lines[3]==1.0 and lines[4]==0.8:{file}')
#         elif lines[2]=='decay_rate: 0.9':
#             print(f'lines[2]==0.9 and lines[3]==1.0 and lines[4]==0.8:{file}')
#     else:
#         continue
