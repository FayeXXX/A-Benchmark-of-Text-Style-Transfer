import torch
from tqdm import tqdm
import pickle
import torch.nn.functional as F
text_list_0 = pickle.load(open('text_list_0', 'rb'))
text_list_1 = pickle.load(open('text_list_1', 'rb'))
tmp_tensor_0 = torch.load('tmp_tensor_0')
tmp_tensor_1 = torch.load('tmp_tensor_1')

sentence_list_A = []
sentence_list_B = []
similar_list = []
for i in tqdm(range(len(tmp_tensor_0))):
    similar_scores = F.cosine_similarity(tmp_tensor_0[i].unsqueeze(0).expand(tmp_tensor_1.size(0), -1), tmp_tensor_1, dim=1)
    one_best = torch.argmax(similar_scores).item()
    # print(similar_scores[one_best])
    # print(text_list_0[i], text_list_1[one_best])
    similar_list.append(similar_scores[one_best].item())
    # if similar_scores[one_best] > 0.85:
    if similar_scores[one_best] > 0.7:
        sentence_list_A.append(text_list_0[i]+"\n")
        sentence_list_B.append(text_list_1[one_best]+"\n")

open("merged_A", "w").writelines(sentence_list_A)
open("merged_B", "w").writelines(sentence_list_B)