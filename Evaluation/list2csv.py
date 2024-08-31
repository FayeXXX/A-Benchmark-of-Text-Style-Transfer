import pandas as pd
import os
import numpy as np
import argparse
import math


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hasref', action='store_true', default=True)
    parser.add_argument('--multiref', action='store_true', default=True)
    parser.add_argument('--algorithm', type=str, default='tyb')
    parser.add_argument('--dataset', type=str, default='styleptb_TFU')
    args = parser.parse_args()

    if args.dataset in ["yelp", "gyafc_fr", "gyafc_em"]:
        args.hasref = True
        args.multiref = True
    elif args.dataset in ["shakespeare", "amazon", "styleptb_TFU", "styleptb_ARR"]:
        args.hasref = True
        args.multiref = False

    col = ['model', 'acc_bert', 's_bleu', 'r_bleu', 'multi_bleu','ppl_kenlm', 'G2self', 'G2ref4', 'G4']
    data_list = []
    eval_dir = f'eval_out/{args.algorithm}/{args.dataset}'
    from pathlib import Path
    path = Path(eval_dir)
    dirs = [e for e in path.iterdir() if e.is_dir()]

    count = 0
    df0 = pd.DataFrame(columns=col)  # 分别为2种风格创建评估结果表格
    df1 = pd.DataFrame(columns=col)
    for dir in dirs:
        for file in os.listdir(f'{dir}'):
            with open(f'{dir}/{file}','r') as f:
                lines = []
                lines.append(dir)
                lines.extend([float(line.strip()) if line.strip()!='None' else line.strip() for line in f.readlines()])
                if len(lines)!=6:
                    print(dir)
                    continue
                #overall score G2,G3,G4
                G2sf = np.sqrt(lines[1] * lines[2])

                if args.hasref:
                    G2rf = np.sqrt(lines[1] * lines[3])
                else:
                    G2rf = None
                if args.multiref:
                    try:
                        G2multi = np.sqrt(lines[1] * lines[4])
                        G3 = math.pow(lines[1] * lines[4] * 1.0 / math.log(lines[5]), 1.0 / 3.0)
                    except:
                        print('error')
                else:
                    G2multi = 'None'
                    G3 = math.pow(lines[1] * lines[3] * 1.0 / math.log(lines[5]), 1.0 / 3.0)
                lines.extend([G2rf, G2multi, G3])
                # lines = np.array(lines).reshape(1, 9)

            if file[3]=='0':
                df0.loc[len(df0.index)] = lines
                count+=1
            else:
                df1.loc[len(df1.index)] = lines
                count += 1

    df0.to_csv(f"./eval_out/{args.algorithm}/{args.dataset}/all_0.csv", encoding='utf-8', mode='a+', index=False, header=True)
    df1.to_csv(f"./eval_out/{args.algorithm}/{args.dataset}/all_1.csv", encoding='utf-8', mode='a+', index=False, header=True)
    print('cunt:',count)