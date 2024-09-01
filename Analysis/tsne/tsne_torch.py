import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
from sklearn.manifold import TSNE
import torch.nn as nn

def mscatter(x, y, ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax = plt.gca()
    sc = ax.scatter(x, y, **kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


if __name__ == "__main__":
    np.random.seed(0)
    parser = argparse.ArgumentParser()
    # parser.add_argument("--file_path", type=str, default="/home/ubuntu/model/Daining-style-transformer-master-2/style-transformer/save/Nov13071842",
    #                     help="file name of feature stored")
    parser.add_argument("--file_path", type=str,
                        default="./embeddings",
                        help="file name of feature stored")

    # parser.add_argument("--file_path", type=str,
    #                     default="/home/ubuntu/projects/tsne-pytorch-master/embeddings",
    #                     help="file name of feature stored")

    opt = parser.parse_args()
    print("get choice from args", opt)
    # iter = [-1, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000] 1200.emb3.npy
    iter = np.arange(100, 8000, 100)
    for i in iter:
        X_pos, X_neg = 0, 0
        # x_pos_file = opt.file_path + f"/{i}.emb.neg.npy"
        # x_neg_file = opt.file_path + f"/{i}.emb.pos.npy"
        x_pos_file = './embeddings/yelp_1_ref_cls.npy'
        x_neg_file = './embeddings/yelp_0_ref_cls.npy'
        #
        X_pos = np.load(x_pos_file)
        X_neg = np.load(x_neg_file)
        # dist = np.matmul(np.mean(X_pos, axis=0), np.mean(X_neg, axis=0))
        # print(dist)
        X_pos = nn.functional.normalize(torch.Tensor(np.load(x_pos_file)), p=2, dim=-1)
        X_neg = nn.functional.normalize(torch.Tensor(np.load(x_neg_file)), p=2, dim=-1)
        sim_00 = torch.mean(torch.matmul(X_pos, X_pos.T))
        sim_11 = torch.mean(torch.matmul(X_neg, X_neg.T))
        sim_01 = torch.mean(torch.matmul(X_pos, X_neg.T))
        print(sim_00, sim_11, sim_01)

        X_pos_label = np.ones(X_pos.shape[0]).tolist()
        X_neg_label = np.zeros(X_neg.shape[0]).tolist()
        X = np.concatenate((X_pos, X_neg), axis=0)
        labels = X_pos_label + X_neg_label
        # labels = np.random.randint(0,1,)
        # confirm that x file get same number point than label file
        # otherwise may cause error in scatter
        assert (len(X[:, 0]) == len(X[:, 1]))
        assert (len(X) == len(labels))

        model = TSNE(n_components=2, perplexity=5, n_iter=270)
        Y = model.fit_transform(X)
        map_color = {1: '#A32A31', 0: '#14507D'}
        colors = list(map(lambda x: map_color[x], labels))
        # map_marker = {-1: 'o', 1: 's'}
        # markers = list(map(lambda x: map_marker[x], labels))
        # mscatter(Y[:, 0], Y[:, 1], s=20, c=color, m=makers)

        # styles = plt.style.available
        # plt.style.use('seaborn-darkgrid')
        scatter = plt.scatter(Y[:, 0], Y[:, 1], 20, c=colors, marker='o', alpha=0.4)
        plt.axis('off')
        # plt.legend(handles=scatter.legend_elements()[0], labels=['pos', 'neg'])
        plt.savefig(opt.file_path + f"/ref_tsne.pdf", bbox_inches='tight')
        plt.show()
