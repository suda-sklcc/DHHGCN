import argparse
import pickle
import time
from model import *
import matplotlib.pyplot as plt
import os
from sklearn.metrics import ndcg_score
from utils import Data,hitRatio, MRR, \
     do_dataset_sparse, EarlyStopping
import pandas as pd
import os.path
import random
from torch.utils.data import DataLoader
import numpy as np
import scipy.sparse as sp
import dgl
from torch import optim
import torch
from tqdm import tqdm
import torch.nn.functional as F
from scipy.sparse import csr_matrix, csc_matrix
from functools import reduce
import numba

torch.cuda.empty_cache()
method_name = 'DHHGCN'
numba.config.NUMBA_DEFAULT_NUM_THREADS = 4
numba.config.NUMBA_NUM_THREADS = 4

file_path='data4/'
datasets_name = ['Clothing_Shoes_and_Jewelry', 'Sports_and_Outdoors']

k_list = [5, 10]
best_ndcg_a = {k: 0 for k in k_list}
best_hit_ratio_a = {k: 0 for k in k_list}
best_ndcg_b = {k: 0 for k in k_list}
best_hit_ratio_b = {k: 0 for k in k_list}
best_mrr_a = {k: 0 for k in k_list}
best_mrr_b = {k: 0 for k in k_list}
this_ndcg_a = {k: 0 for k in k_list}
this_ndcg_b = {k: 0 for k in k_list}
all_results = [[], [], [], []]
train_loss_list = []
test_loss_list = []

SPLIT = "=" * 50

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--k-single', type=int, default=2, help='K value of hypergraph for single domain')
parser.add_argument('--k-dual', type=int, default=5, help='K value of hypergraph for dual domain')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight-decay', type=float, default=0.00001, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dataset', type=str, default="Amazon", help="Training in which dataset")
parser.add_argument('--batch-size', type=int, default=100, help="The batch size of training")
parser.add_argument('--single-layer-num', type=int, default=2, help="The conv layer num of single domain")
parser.add_argument('--dual-layer-num', type=int, default=1, help="The conv layer num of dual domain")
parser.add_argument('--t_percent', type=float, default=1.0, help='target percent')
parser.add_argument('--s_percent', type=float, default=1.0, help='source percent')
parser.add_argument('--pos-weight', type=float, default=1.0, help='weight for positive samples')
parser.add_argument('--embedding-size', type=int, default=128, help='embedding size')
parser.add_argument('--neg-frequency', type=int, default=5, help='negative sample choice frequency')
parser.add_argument('--if-sparse', type=bool, default=False, help='if doing sparse experiment')
parser.add_argument('--sparse-ratio', type=int, default=30, help='the sparse ratio of our experiment')
parser.add_argument('--log', type=str, default='logs/{}'.format(method_name), help='log directory')
parser.add_argument('--cuda-index', type=int, default=0, help='train in which GPU')
parser.add_argument('--model-structure', type=str, default="single", help='normal,single')
parser.add_argument('--graph-type', type=str, default="heterogeneous", help='homogeneous,heterogeneous')
parser.add_argument('--intra-type', type=str, default="_top05", help="the construction principle of intra-domain hypergraph")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Train in {device}")



print(f"Train in {args.dataset}, epoch = {args.epochs}")
print(args)

log = os.path.join(args.log,
                   '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(args.dataset, args.embedding_size, args.epochs, args.lr,
                                                                args.weight_decay, args.k_single, args.single_layer_num,
                                                                args.dual_layer_num,
                                                                args.k_dual, args.if_sparse,
                                                                args.sparse_ratio,
                                                                args.intra_type))
if os.path.isdir(log):
    print("%s already exist. are you sure to override? Ok, I'll wait for 5 seconds. Ctrl-C to abort." % log)
    time.sleep(5)
    os.system('rm -rf %s/' % log)

os.makedirs(log, exist_ok=True)
print("made the log directory", log)

if args.if_sparse:
    print(f"Training in sparse dataset, sparse ratio = {args.sparse_ratio}")
    file_path = os.path.join(file_path,f"sparse_{args.sparse_ratio}/")
else:
    print(f"Training in no sparse dataset")


#original
a_node = 4250
a_price = 10
a_category = 523

A_HGtrain = pickle.load(open(file_path+datasets_name[0]+'_HGtrain.txt', 'rb'))
A_HGtrain = Data(A_HGtrain, shuffle=True, n_node=a_node, n_price=a_price, n_category=a_category)

#original
b_node = 8096
b_price = 10
b_category = 919


B_HGtrain = pickle.load(open(file_path+datasets_name[1]+'_HGtrain.txt', 'rb'))
B_HGtrain = Data(B_HGtrain, shuffle=True, n_node=b_node, n_price=b_price, n_category=b_category)

#original
NUM_USER_CLOTH =  3468
NUM_USER_SPORTS = 3468


USER_OVERLAP_A = pd.read_csv('intersection_AB_'+datasets_name[0]+'.txt', sep=' ')['userID']
USER_OVERLAP_B = pd.read_csv('intersection_AB_'+datasets_name[1]+'.txt', sep=' ')['userID']


single_back = f"_{args.k_single}" + args.intra_type + ".npz"
print(f"back = {single_back}")


conv_au = np.load(os.path.join(file_path, "conv_"+datasets_name[0] + single_back))["arr_0"]
conv_bu = np.load(os.path.join(file_path, "conv_"+datasets_name[1] + single_back))["arr_0"]
HyGCN_a_u = np.load(os.path.join(file_path, "Ha_GCN_u.npz"))["arr_0"]
HyGCN_b_u = np.load(os.path.join(file_path, "Hb_GCN_u.npz"))["arr_0"]
HyGCN_a_i = np.load(os.path.join(file_path, "Ha_GCN_i.npz"))["arr_0"]
HyGCN_b_i = np.load(os.path.join(file_path, "Hb_GCN_i.npz"))["arr_0"]


HyGCN_a_i = torch.FloatTensor(HyGCN_a_i).to(device)
HyGCN_b_i = torch.FloatTensor(HyGCN_b_i).to(device)
conv_au = torch.FloatTensor(conv_au).to(device)
conv_bu = torch.FloatTensor(conv_bu).to(device)
HyGCN_a_u = torch.FloatTensor(HyGCN_a_u).to(device)
HyGCN_b_u = torch.FloatTensor(HyGCN_b_u).to(device)
USER_OVERLAP_A = torch.LongTensor(USER_OVERLAP_A.to_numpy()).to(device)
USER_OVERLAP_B = torch.LongTensor(USER_OVERLAP_B.to_numpy()).to(device)

model = DHHGCN(embedding_size=args.embedding_size, user_overlap_a=USER_OVERLAP_A, user_overlap_b=USER_OVERLAP_B,
               user_num_a=NUM_USER_CLOTH, item_num_a=a_node,
               user_num_b=NUM_USER_SPORTS, item_num_b=b_node,
               device=device,single_layer_num=args.single_layer_num,
               dual_layer_num=args.dual_layer_num, model_structure=args.model_structure,graph_type=args.graph_type,
               A_adjacency=A_HGtrain.adjacency, A_adjacency_pp=A_HGtrain.adjacency_pp,
               A_adjacency_cc=A_HGtrain.adjacency_cc,
               A_adjacency_vp=A_HGtrain.adjacency_vp, A_adjacency_vc=A_HGtrain.adjacency_vc,
               A_adjacency_pv=A_HGtrain.adjacency_pv, A_adjacency_pc=A_HGtrain.adjacency_pc,
               A_adjacency_cv=A_HGtrain.adjacency_cv, A_adjacency_cp=A_HGtrain.adjacency_cp,
               a_node=a_node, a_price=a_price, a_category=a_category,
              B_adjacency=B_HGtrain.adjacency, B_adjacency_pp=B_HGtrain.adjacency_pp,
              B_adjacency_cc=B_HGtrain.adjacency_cc,
              B_adjacency_vp=B_HGtrain.adjacency_vp, B_adjacency_vc=B_HGtrain.adjacency_vc,
              B_adjacency_pv=B_HGtrain.adjacency_pv, B_adjacency_pc=B_HGtrain.adjacency_pc,
              B_adjacency_cv=B_HGtrain.adjacency_cv, B_adjacency_cp=B_HGtrain.adjacency_cp,
              b_node=b_node, b_price=b_price, b_category=b_category,HyGCN_a_i=HyGCN_a_i, HyGCN_b_i=HyGCN_b_i,
              HyGCN_a_u=HyGCN_a_u,HyGCN_b_u=HyGCN_b_u,conv_au= conv_au,conv_bu=conv_bu)

if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model=torch.nn.DataParallel(model)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
loss_func = F.binary_cross_entropy

early_stop = EarlyStopping()

user_neg_sample_cloth_dict = {}
user_neg_sample_sports_dict = {}
neg_sample_length_cloth = {}
neg_sample_length_sports = {}

peo2cloth=pd.read_csv(file_path+datasets_name[0]+'.txt', sep=' ')
peo2sports=pd.read_csv(file_path+datasets_name[1]+'.txt', sep=' ')

cloth_nega=pd.read_csv(file_path+datasets_name[0]+'_nega.txt',sep=' ')
sports_nega=pd.read_csv(file_path+datasets_name[1]+'_nega.txt',sep=' ')

cloth_test=pd.read_csv(file_path+datasets_name[0]+'_test.txt',sep=' ')
sports_test=pd.read_csv(file_path+datasets_name[1]+'_test.txt',sep=' ')

cloth_vali=pd.read_csv(file_path+datasets_name[0]+'_vali.txt',sep=' ')
sports_vali=pd.read_csv(file_path+datasets_name[1]+'_vali.txt',sep=' ')

cloth_set =pd.read_csv(file_path+datasets_name[0]+'_item.txt',sep=' ')['itemID']
sports_set =pd.read_csv(file_path+datasets_name[1]+'_item.txt',sep=' ')['itemID']


def remove_test_and_vali(peo2item, test, vali):
    for i in tqdm(range(len(peo2item['user']))):
        item_list = eval(peo2item['itemID'][i])
        test_item_id = test['itemID'][i]
        vali_item_id = vali['itemID'][i]

        if test_item_id in item_list:
            item_list.remove(test_item_id)

        if vali_item_id in item_list:
            item_list.remove(vali_item_id)

        peo2item['itemID'][i] = str(item_list)

    return peo2item
peo2cloth= remove_test_and_vali(peo2cloth,cloth_test,cloth_vali)
peo2sports= remove_test_and_vali(peo2sports,sports_test,sports_vali)

for user in tqdm(peo2cloth['userID']):
    user_neg_sample_cloth_dict[user] = list(set(cloth_set) - set(eval(peo2cloth['itemID'][user])) - set(cloth_nega['1'][user]))
    neg_sample_length_cloth[user] = len(eval(peo2cloth['itemID'][user]))

for user in tqdm(peo2sports['userID']):
    user_neg_sample_sports_dict[user] = list(set(sports_set) - set(eval(peo2sports['itemID'][user])) - set(sports_nega['1'][user]))
    neg_sample_length_sports[user] = len(eval(peo2sports['itemID'][user]))
print("Neg sample prepare succeed")


user_id_cloth = np.arange(NUM_USER_CLOTH).reshape([NUM_USER_CLOTH, 1])
user_id_sports = np.arange(NUM_USER_SPORTS).reshape([NUM_USER_SPORTS, 1])

train_loader_cloth = torch.utils.data.DataLoader(torch.from_numpy(user_id_cloth),batch_size=args.batch_size,shuffle=True)
train_loader_sports = torch.utils.data.DataLoader(torch.from_numpy(user_id_sports), batch_size=args.batch_size,shuffle=True)

save_loader_cloth=torch.utils.data.DataLoader(torch.from_numpy(user_id_cloth),batch_size=args.batch_size,shuffle=False)
save_loader_sports=torch.utils.data.DataLoader(torch.from_numpy(user_id_sports),batch_size=args.batch_size,shuffle=False)


user_neg_sample_sports = {}
user_neg_sample_cloth = {}

def neg_sample():
    t = time.time()
    user_neg_sample_a = {}
    user_neg_sample_b = {}
    for user_name in tqdm(peo2cloth['userID']):
        user_neg_sample_a[user_name] = np.random.choice(user_neg_sample_cloth_dict[user_name],neg_sample_length_cloth[user_name], replace=False)
    for user_name in tqdm(peo2sports['userID']):
        user_neg_sample_b[user_name] = np.random.choice(user_neg_sample_sports_dict[user_name],neg_sample_length_sports[user_name], replace=False)
    print(f"Sample succeed. Time = {time.time() - t}")
    return user_neg_sample_a, user_neg_sample_b

def load_batch_train_sample(users1,users2):
    users1 = np.array(users1).squeeze()
    users2 = np.array(users2).squeeze()
    user_list_cloth = []
    user_list_sports = []

    pos_result_cloth = []
    neg_result_cloth = []
    pos_result_sports = []
    neg_result_sports = []
    for user in users1:
        user_list_cloth.extend([user for _ in range(len(eval(peo2cloth['itemID'][user])))])
        pos_result_cloth.extend(eval(peo2cloth['itemID'][user]))
        neg_result_cloth.extend(user_neg_sample_cloth[user])
    for user in users2:
        user_list_sports.extend([user for _ in range(len(eval(peo2sports['itemID'][user])))])
        pos_result_sports.extend(eval(peo2sports['itemID'][user]))
        neg_result_sports.extend(user_neg_sample_sports[user])
    return user_list_cloth, user_list_sports, pos_result_cloth, pos_result_sports, neg_result_cloth, neg_result_sports

def load_batch_test_sample():
    user_list_cloth = []
    user_list_sports = []
    item_sample_cloth = []
    item_sample_sports = []
    true_label_cloth = []
    true_label_sports = []
    neg_99 = [0 for _ in range(99)]
    for user in peo2cloth['userID']:
        user_list_cloth.extend([user for _ in range(100)])
        item_sample_cloth.append(cloth_test['itemID'][user])
        item_sample_cloth.extend(eval(cloth_nega['1'][user]))
        true_label_cloth.append(1)
        true_label_cloth.extend(neg_99)

    for user in peo2sports['userID']:
        user_list_sports.extend([user for _ in range(100)])
        item_sample_sports.append(sports_test['itemID'][user])
        item_sample_sports.extend(eval(sports_nega['1'][user]))
        true_label_sports.append(1)
        true_label_sports.extend(neg_99)

    return user_list_cloth, user_list_sports, item_sample_cloth, item_sample_sports, true_label_cloth, true_label_sports

def train(epoch):
    torch.autograd.set_detect_anomaly(True)
    print(SPLIT)
    print(f"epoch: {epoch}")
    model.train()
    loss_list = []
    loss_a_list = []
    loss_b_list = []

    for (batch_idx_cloth, data_cloth), (batch_idx_sports, data_sports) in tqdm(zip(enumerate(train_loader_cloth), enumerate(train_loader_sports))):
        data_cloth = data_cloth.reshape([-1])
        data_sports = data_sports.reshape([-1])

        user_list_cloth,user_list_sports, pos_sample_cloth, pos_sample_sports,neg_sample_cloth,neg_sample_sports= load_batch_train_sample(data_cloth,data_sports)


        user_list_cloth = torch.LongTensor(user_list_cloth).to(device)
        user_list_sports = torch.LongTensor(user_list_sports).to(device)

        pos_sample_cloth = torch.LongTensor(pos_sample_cloth).to(device)
        neg_sample_cloth = torch.LongTensor(neg_sample_cloth).to(device)
        pos_sample_sports = torch.LongTensor(pos_sample_sports).to(device)
        neg_sample_sports = torch.LongTensor(neg_sample_sports).to(device)

        model.train()
        optimizer.zero_grad()

        pos_score_a, pos_score_b = model.forward(user_list_cloth, user_list_sports,pos_sample_cloth,pos_sample_sports)

        neg_score_a, neg_score_b = model.forward(user_list_cloth,user_list_sports, neg_sample_cloth,neg_sample_sports)

        predict_label_a = torch.cat((pos_score_a, neg_score_a))
        true_label_a = torch.cat((torch.ones_like(pos_score_a), torch.zeros_like(neg_score_a)))
        predict_label_b = torch.cat((pos_score_b, neg_score_b))
        true_label_b = torch.cat((torch.ones_like(pos_score_b), torch.zeros_like(neg_score_b)))


        loss_a = loss_func(predict_label_a, true_label_a)
        loss_b = loss_func(predict_label_b, true_label_b)
        loss_a_list.append(loss_a.item())
        loss_b_list.append(loss_b.item())


        loss = torch.add(loss_a, loss_b)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

    print(f"Loss = {np.mean(loss_list)}, "
          f"Loss in domain a = {np.mean(loss_a_list)}, "
          f"Loss in domain b = {np.mean(loss_b_list)}")
    train_loss_list.append([np.mean(loss_list)])


def load_batch_vali_sample():
    user_list_cloth = []
    user_list_sports = []
    item_sample_cloth = []
    item_sample_sports = []
    true_label_cloth = []
    true_label_sports = []
    neg_99 = [0 for _ in range(99)]
    for user in peo2cloth['userID']:
        user_list_cloth.extend([user for _ in range(100)])
        item_sample_cloth.append(cloth_vali['itemID'][user])
        item_sample_cloth.extend(eval(cloth_nega['1'][user]))
        true_label_cloth.append(1)
        true_label_cloth.extend(neg_99)

    for user in peo2sports['userID']:
        user_list_sports.extend([user for _ in range(100)])
        item_sample_sports.append(sports_vali['itemID'][user])
        item_sample_sports.extend(eval(sports_nega['1'][user]))
        true_label_sports.append(1)
        true_label_sports.extend(neg_99)
    return user_list_cloth, user_list_sports, item_sample_cloth, item_sample_sports, true_label_cloth, true_label_sports

@torch.no_grad()
def test(best_ndcg_a, best_hit_ratio_a, best_ndcg_b, best_hit_ratio_b, best_mrr_a, best_mrr_b, all_results: list,
         this_ndcg_a, this_ndcg_b):
    user_list_cloth, user_list_sports, item_sample_cloth, item_sample_sports, true_label_cloth, true_label_sports \
        = load_batch_vali_sample()
    # if args.cuda:
    user_list_cloth = torch.LongTensor(user_list_cloth).to(device)
    user_list_sports = torch.LongTensor(user_list_sports).to(device)
    item_sample_cloth = torch.LongTensor(item_sample_cloth).to(device)
    item_sample_sports = torch.LongTensor(item_sample_sports).to(device)
    true_label_cloth = torch.FloatTensor(true_label_cloth).to(device)
    true_label_sports = torch.FloatTensor(true_label_sports).to(device)
    pred_score_a, pred_score_b = model.forward(user_list_cloth, user_list_sports, item_sample_cloth,item_sample_sports,)
    true_label_a = np.array([np.concatenate((np.ones(1), np.zeros(99))) for _ in cloth_test['userID']])
    true_label_b = np.array([np.concatenate((np.ones(1), np.zeros(99))) for _ in sports_test['userID']])
    pred_label_a = pred_score_a.cpu().numpy().reshape(-1, 100)
    pred_label_b = pred_score_b.cpu().numpy().reshape(-1, 100)
    print(SPLIT)
    print("Begin Test!!!")
    loss_a = loss_func(pred_score_a, true_label_cloth)
    loss_b = loss_func(pred_score_b, true_label_sports)
    loss = torch.add(loss_a, loss_b)
    print(f"Test loss = {loss}, Domain A loss = {loss_a}, Domain B loss = {loss_b}")
    test_loss_list.append([loss.item()])
    print(f"NDCG@K for Domain A:")
    for k in k_list:
        ndcg_a = round(ndcg_score(y_true=true_label_a, y_score=pred_label_a, k=k), 4)
        this_ndcg_a[k] = ndcg_a
        if ndcg_a > best_ndcg_a[k]:
            best_ndcg_a[k] = ndcg_a
            torch.save(model.module.state_dict(), os.path.join(log, f'best_ndcg_a_{k}.pkl'))
        print(f"k:{k}, ndcg = {ndcg_a}", end="\t\t")
    print()
    print(f"NDCG@K for Domain B:")
    for k in k_list:
        ndcg_b = round(ndcg_score(y_true=true_label_b, y_score=pred_label_b, k=k), 4)
        this_ndcg_b[k] = ndcg_b
        if ndcg_b > best_ndcg_b[k]:
            best_ndcg_b[k] = ndcg_b
            torch.save(model.module.state_dict(), os.path.join(log, f'best_ndcg_b_{k}.pkl'))
        print(f"k:{k}, ndcg = {ndcg_b}", end="\t\t")
    print()
    print(f"MRR@K for Domain A:")
    for k in k_list:
        mrr_a = round(MRR(pred_score_a.reshape(-1, 100), k=k), 4)
        if mrr_a > best_mrr_a[k]:
            best_mrr_a[k] = mrr_a
            torch.save(model.module.state_dict(), os.path.join(log, f'best_mrr_a_{k}.pkl'))
        print(f"k:{k}, mrr = {mrr_a}", end="\t\t")
    print()
    print(f"MRR@K for Domain B:")
    for k in k_list:
        mrr_b = round(MRR(pred_score_b.reshape(-1, 100), k=k), 4)
        if mrr_b > best_mrr_b[k]:
            best_mrr_b[k] = mrr_b
            torch.save(model.module.state_dict(), os.path.join(log, f'best_mrr_b_{k}.pkl'))
        print(f"k:{k}, mrr = {mrr_b}", end="\t\t")
    print()
    print(f"HitRatio@K for Domain A: ")
    for k in k_list:
        hit_a = round(hitRatio(pred_score_a.reshape(-1, 100), k), 4)
        if hit_a > best_hit_ratio_a[k]:
            best_hit_ratio_a[k] = hit_a
            torch.save(model.module.state_dict(), os.path.join(log, f'best_hit_a_{k}.pkl'))
        print(f"k:{k}, hitRatio = {hit_a}", end="\t")
    print()
    print(f"HitRatio@K for Domain B: ")
    for k in k_list:
        hit_b = round(hitRatio(pred_score_b.reshape(-1, 100), k), 4)
        if hit_b > best_hit_ratio_b[k]:
            best_hit_ratio_b[k] = hit_b
            torch.save(model.module.state_dict(), os.path.join(log, f'best_hit_b_{k}.pkl'))
        print(f"k:{k}, hitRatio = {hit_b}", end="\t")
    print()
    print(f"Best Results")
    print(f"Domain A:")
    for k in k_list:
        print(f"K:{k}, ndcg = {best_ndcg_a[k]}, hitRatio = {best_hit_ratio_a[k]}, mrr = {best_mrr_a[k]}")
    print(f"Domain B:")
    for k in k_list:
        print(f"K:{k}, ndcg = {best_ndcg_b[k]}, hitRatio = {best_hit_ratio_b[k]}, mrr = {best_mrr_b[k]}")
    print(SPLIT)

@torch.no_grad()
def test_best_result() -> object:
    user_list_cloth, user_list_sports, item_sample_cloth, item_sample_sports, true_label_cloth, true_label_sports = load_batch_test_sample()
    # if args.cuda:
    user_list_cloth = torch.LongTensor(user_list_cloth).to(device)
    user_list_sports = torch.LongTensor(user_list_sports).to(device)
    item_sample_cloth = torch.LongTensor(item_sample_cloth).to(device)
    item_sample_sports = torch.LongTensor(item_sample_sports).to(device)
    true_label_a = np.array([np.concatenate((np.ones(1), np.zeros(99))) for _ in cloth_test['userID']])
    true_label_b = np.array([np.concatenate((np.ones(1), np.zeros(99))) for _ in sports_test['userID']])
    for k in k_list:
        # ndcg a
        state_dict = torch.load(os.path.join(log, f'best_ndcg_a_{k}.pkl'))
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.module.load_state_dict(state_dict, strict=False)

        pred_score_a, pred_score_b = model.forward(user_list_cloth, user_list_sports, item_sample_cloth,item_sample_sports)
        pred_label_a = pred_score_a.cpu().numpy().reshape(-1, 100)
        ndcg_a = round(ndcg_score(y_true=true_label_a, y_score=pred_label_a, k=k), 4)

        # ndcg b
        state_dict = torch.load(os.path.join(log, f'best_ndcg_b_{k}.pkl'))
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.module.load_state_dict(state_dict, strict=False)
        pred_score_a, pred_score_b = model.forward(user_list_cloth, user_list_sports, item_sample_cloth,item_sample_sports)
        pred_label_b = pred_score_b.cpu().numpy().reshape(-1, 100)
        ndcg_b = round(ndcg_score(y_true=true_label_b, y_score=pred_label_b, k=k), 4)

        # hit a
        state_dict = torch.load(os.path.join(log, f'best_hit_a_{k}.pkl'))
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.module.load_state_dict(state_dict)
        pred_score_a, pred_score_b = model.forward(user_list_cloth, user_list_sports, item_sample_cloth,item_sample_sports)
        hit_a = round(hitRatio(pred_score_a.reshape(-1, 100), k), 4)

        # hit b
        state_dict = torch.load(os.path.join(log, f'best_hit_b_{k}.pkl'))
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.module.load_state_dict(state_dict)
        pred_score_a, pred_score_b = model.forward(user_list_cloth, user_list_sports, item_sample_cloth,item_sample_sports)
        hit_b = round(hitRatio(pred_score_b.reshape(-1, 100), k), 4)

        # mrr a
        state_dict = torch.load(os.path.join(log, f'best_mrr_a_{k}.pkl'))
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.module.load_state_dict(state_dict)
        pred_score_a, pred_score_b = model.forward(user_list_cloth, user_list_sports, item_sample_cloth,item_sample_sports)
        mrr_a = round(MRR(pred_score_a.reshape(-1, 100), k=k), 4)

        # mrr b
        state_dict = torch.load(os.path.join(log, f'best_mrr_b_{k}.pkl'))
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.module.load_state_dict(state_dict)
        pred_score_a, pred_score_b = model.forward(user_list_cloth, user_list_sports, item_sample_cloth,item_sample_sports)
        mrr_b = round(MRR(pred_score_b.reshape(-1, 100), k=k), 4)

        print(f'Test TopK:{k} ---> cloth: hr:{hit_a:.4f}, ndcg:{ndcg_a:.4f}, mrr:{mrr_a:.4f}, sports: hr:{hit_b:.4f}, ndcg:{ndcg_b:.4f}, mrr:{mrr_b:.4f}')



for epoch in range(args.epochs):

    if epoch < 20:
        if epoch % 10 == 0:
            print("Do negative sampling")
            user_neg_sample_cloth, user_neg_sample_sports = neg_sample()
    elif epoch < 40:
        if epoch % 5 == 0:
            print("Do negative sampling")
            user_neg_sample_cloth, user_neg_sample_sports = neg_sample()
    else:
        print("Do negative sampling")
        user_neg_sample_cloth, user_neg_sample_sports = neg_sample()
    train(epoch)
    torch.cuda.empty_cache()
    test(best_ndcg_a, best_hit_ratio_a, best_ndcg_b, best_hit_ratio_b, best_mrr_a, best_mrr_b, all_results, this_ndcg_a,
         this_ndcg_b)
    if epoch > 40 and not early_stop.update(epoch, this_ndcg_a[5], this_ndcg_b[5], this_ndcg_a[10], this_ndcg_b[10]):
        print(f"Best epoch get, epoch = {epoch}")
        print(SPLIT)
        break
    if (epoch + 1) % 20== 0:
        test_best_result()
test_best_result()

