
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
import scipy.sparse as sp

from tqdm import tqdm
import torch

def data_easy_masks(data_l, n_row, n_col):
    data, indices, indptr  = data_l[0], data_l[1], data_l[2]
    matrix = csr_matrix((data, indices, indptr), shape=(n_row, n_col))
    return matrix

class Data():
    def __init__(self, data, shuffle=False, n_node=None, n_price=None, n_category=None):
        self.raw = np.asarray(data[0], dtype=object)
        self.price_raw = np.asarray(data[1], dtype=object)

        H_T = data_easy_masks(data[2], len(data[0]), n_node)
        BH_T = H_T.T.multiply(1.0 / H_T.sum(axis=1).reshape(1, -1))
        BH_T = BH_T.T
        H = H_T.T
        DH = H.T.multiply(1.0 / H.sum(axis=1).reshape(1, -1))
        DH = DH.T
        DHBH_T = np.dot(DH, BH_T)

        H_p_T = data_easy_masks(data[3], len(data[0]), n_price)
        BH_p_T = H_p_T.T.multiply(1.0 / H_p_T.sum(axis=1).reshape(1, -1))
        BH_p_T = BH_p_T.T
        H_p = H_p_T.T
        DH_p = H_p.T.multiply(1.0 / H_p.sum(axis=1).reshape(1, -1))
        DH_p = DH_p.T
        DHBH_p_T = np.dot(DH_p, BH_p_T)


        H_c_T = data_easy_masks(data[4], len(data[0]), n_category)
        BH_c_T = H_c_T.T.multiply(1.0 / H_c_T.sum(axis=1).reshape(1, -1))
        BH_c_T = BH_c_T.T
        H_c = H_c_T.T
        DH_c = H_c.T.multiply(1.0 / H_c.sum(axis=1).reshape(1, -1))
        DH_c = DH_c.T
        DHBH_c_T = np.dot(DH_c, BH_c_T)


        H_pv = data_easy_masks(data[5], n_price, n_node)
        BH_pv = H_pv
        BH_vp = H_pv.T

        H_pc = data_easy_masks(data[6], n_price, n_category)
        BH_pc = H_pc
        BH_cp = H_pc.T

        H_cv = data_easy_masks(data[7], n_category, n_node)
        BH_cv = H_cv
        BH_vc = H_cv.T

        self.adjacency = DHBH_T.tocoo()
        self.adjacency_pp = DHBH_p_T.tocoo()
        self.adjacency_cc = DHBH_c_T.tocoo()

        self.adjacency_pv = BH_pv.tocoo()
        self.adjacency_pc = BH_pc.tocoo()

        self.adjacency_vp = BH_vp.tocoo()
        self.adjacency_vc = BH_vc.tocoo()


        self.adjacency_cp = BH_cp.tocoo()
        self.adjacency_cv = BH_cv.tocoo()

        self.n_node = n_node
        self.n_price = n_price
        self.n_category = n_category
        self.length = len(self.raw)
        self.shuffle = shuffle

class EarlyStopping:

    def __init__(self) -> None:
        self.best_epoch = -1
        self.best_ndcg_a_5 = -1
        self.best_ndcg_b_5 = -1
        self.best_ndcg_a_10 = -1
        self.best_ndcg_b_10 = -1
        self.patience = 10

    def update(self, epoch: int, test_ndcg_a_5, test_ndcg_b_5, test_ndcg_a_10, test_ndcg_b_10):
        if test_ndcg_a_5 > self.best_ndcg_a_5 or test_ndcg_b_5 > self.best_ndcg_b_5 \
                or test_ndcg_a_10 > self.best_ndcg_a_10 or test_ndcg_b_10 > self.best_ndcg_b_10:
            self.best_ndcg_a_5 = max(self.best_ndcg_a_5, test_ndcg_a_5)
            self.best_ndcg_b_5 = max(self.best_ndcg_b_5, test_ndcg_b_5)
            self.best_ndcg_a_10 = max(self.best_ndcg_a_10, test_ndcg_a_10)
            self.best_ndcg_b_10 = max(self.best_ndcg_b_10, test_ndcg_b_10)
            self.best_epoch = epoch
            return 1
        else:
            if epoch - self.best_epoch > self.patience:
                return 0
            else:
                return 1

def hitRatio(pred_label, k):
    hit = 0
    values, indices = torch.topk(pred_label, k)
    indices = indices.cpu().numpy()
    for i in range(pred_label.shape[0]):
        if 0 in indices[i]:
            hit += 1
    return hit / pred_label.shape[0]

def MRR(pred_label, k):
    mrr = 0
    values, indices = torch.topk(pred_label, k)
    indices = indices.cpu().numpy()
    for i in range(pred_label.shape[0]):
        for j in range(len(indices[i])):
            if indices[i][j] == 0:
                mrr += 1 / (j + 1)
                break
    return mrr / pred_label.shape[0]

def do_dataset_sparse(dataset, Ha, Hb):
    for k, v in tqdm(dataset.peo2instrument.items()):
        for item in v:
            if Ha[k][item] == 0:
                dataset.peo2instrument[k].remove(item)
    for k, v in tqdm(dataset.peo2music.items()):
        for item in v:
            if Hb[k][item] == 0:
                dataset.peo2music[k].remove(item)



