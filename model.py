import datetime
import math
import numpy as np
import torch
from torch import nn, backends
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse
from scipy.sparse import coo
import time
from functools import reduce


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


class User_SingleDomain(nn.Module):
    def __init__(self, embedding_size: int, user_num: int, device, layer_num: int, model_structure: str):
        super(User_SingleDomain, self).__init__()
        self.user_emb = nn.Embedding(user_num, embedding_size)
        self.user_index_tensor = torch.LongTensor(list(range(user_num))).to(device)

        self.user_num = user_num
        self.layer_num = layer_num
        self.model_structure = model_structure
        self.linears = nn.ModuleList()
        for _ in range(layer_num):
            self.linears.append(nn.Linear(embedding_size, embedding_size))
        self.predict_layer_u = nn.Sequential(
            nn.Linear(embedding_size * (layer_num + 1), embedding_size),
            nn.ReLU()
        )
        self.predict_layer_i = nn.Sequential(
            nn.Linear(embedding_size * (layer_num + 1), embedding_size),
            nn.ReLU()
        )
        self.predict_test_layer = nn.Sequential(
            nn.Linear(embedding_size, 1),
            nn.Softmax()
        )

    def forward(self,device):
        Eu = [self.user_emb(self.user_index_tensor.to(device))]
        return Eu



class DHHGCN(nn.Module):
    def __init__(self, embedding_size: int, user_overlap_a, user_overlap_b, user_num_a, item_num_a, user_num_b,
                 item_num_b, device, single_layer_num,
                 dual_layer_num, model_structure: str,graph_type:str,
                 A_adjacency, A_adjacency_pp, A_adjacency_cc, A_adjacency_vp, A_adjacency_vc,
                 A_adjacency_pv, A_adjacency_pc, A_adjacency_cv, A_adjacency_cp, a_node, a_price, a_category,
                 B_adjacency, B_adjacency_pp, B_adjacency_cc, B_adjacency_vp, B_adjacency_vc,
                 B_adjacency_pv, B_adjacency_pc, B_adjacency_cv, B_adjacency_cp, b_node, b_price, b_category,HyGCN_a_i, HyGCN_b_i,HyGCN_a_u,HyGCN_b_u,
                 conv_au,conv_bu):
        super(DHHGCN, self).__init__()
        print(f"Train in {model_structure} hypergraph!")
        print(f"Train in {graph_type} hypergraph!")
        self.embedding_size = embedding_size
        self.user_num_a = user_num_a
        self.item_num_a = item_num_a
        self.user_num_b = user_num_b
        self.item_num_b = item_num_b
        self.common_user_num = len(user_overlap_a)
        self.all_user_num=user_num_a+user_num_b-len(user_overlap_a)
        self.single_layer_num = single_layer_num
        self.dual_layer_num = dual_layer_num
        self.model_structure = model_structure
        self.graph_type = graph_type
        self.shared_dim=64

        self.a_node = a_node
        self.b_node = b_node
        self.a_price = a_price
        self.b_price = b_price
        self.a_category = a_category
        self.b_category = b_category
        self.A_adjacency = A_adjacency
        self.B_adjacency = B_adjacency
        self.A_adjacency_pp = A_adjacency_pp
        self.B_adjacency_pp = B_adjacency_pp
        self.A_adjacency_cc = A_adjacency_cc
        self.B_adjacency_cc = B_adjacency_cc

        self.A_adjacency_vp = A_adjacency_vp
        self.A_adjacency_vc = A_adjacency_vc
        self.B_adjacency_vp = B_adjacency_vp
        self.B_adjacency_vc = B_adjacency_vc

        self.A_adjacency_pv = A_adjacency_pv
        self.A_adjacency_pc = A_adjacency_pc
        self.B_adjacency_pv = B_adjacency_pv
        self.B_adjacency_pc = B_adjacency_pc

        self.A_adjacency_cv = A_adjacency_cv
        self.A_adjacency_cp = A_adjacency_cp
        self.B_adjacency_cv = B_adjacency_cv
        self.B_adjacency_cp = B_adjacency_cp

        self.A_embedding = nn.Embedding(self.a_node, self.embedding_size)
        self.B_embedding = nn.Embedding(self.b_node, self.embedding_size)
        self.A_price_embedding = nn.Embedding(self.a_price, self.embedding_size)
        self.B_price_embedding = nn.Embedding(self.b_price, self.embedding_size)
        self.A_category_embedding = nn.Embedding(self.a_category, self.embedding_size)
        self.B_category_embedding = nn.Embedding(self.b_category, self.embedding_size)

        self.device=device
        self.domainA_user_embedding = User_SingleDomain(self.embedding_size, self.user_num_a, self.device, self.single_layer_num, self.model_structure).to(self.device)
        self.domainB_user_embedding = User_SingleDomain(self.embedding_size, self.user_num_b, self.device, self.single_layer_num, self.model_structure).to(self.device)
        self.loss_function = nn.CrossEntropyLoss()
        self.init_parameters()

        self.param_size = self.single_layer_num + 1
        self.ac_func = nn.ReLU()

        self.linears_a = nn.ModuleList()
        self.linears_b = nn.ModuleList()
        for _ in range(self.single_layer_num):
            self.linears_a.append(nn.Linear(embedding_size, embedding_size))
            self.linears_b.append(nn.Linear(embedding_size, embedding_size))

        self.tran_pv = nn.Linear(self.embedding_size, self.embedding_size)
        self.tran_pc = nn.Linear(self.embedding_size, self.embedding_size)

        self.tran_cv = nn.Linear(self.embedding_size, self.embedding_size)
        self.tran_cp = nn.Linear(self.embedding_size, self.embedding_size)

        self.w_v_1 = nn.Linear(self.embedding_size * 3, self.embedding_size)
        self.w_v_11 = nn.Linear(self.embedding_size * 2, self.embedding_size)
        self.w_v_2 = nn.Linear(self.embedding_size * 1, self.embedding_size)
        self.w_v_3 = nn.Linear(self.embedding_size * 1, self.embedding_size)

        self.w_p_1 = nn.Linear(self.embedding_size * 3, self.embedding_size)
        self.w_p_11 = nn.Linear(self.embedding_size * 2, self.embedding_size)
        self.w_p_2 = nn.Linear(self.embedding_size * 1, self.embedding_size)
        self.w_p_3 = nn.Linear(self.embedding_size * 1, self.embedding_size)

        self.w_c_1 = nn.Linear(self.embedding_size * 3, self.embedding_size)
        self.w_c_11 = nn.Linear(self.embedding_size * 2, self.embedding_size)
        self.w_c_2 = nn.Linear(self.embedding_size * 1, self.embedding_size)
        self.w_c_3 = nn.Linear(self.embedding_size * 1, self.embedding_size)

        self.HyGCN_a_i=HyGCN_a_i
        self.HyGCN_b_i=HyGCN_b_i
        self.HyGCN_a_u=HyGCN_a_u
        self.HyGCN_b_u=HyGCN_b_u
        self.conv_au=conv_au
        self.conv_bu=conv_bu

        self.user_overlap_a=user_overlap_a
        self.user_overlap_b=user_overlap_b

        self.gate_a = nn.Linear(128, 128)
        self.gate_b = nn.Linear(128, 128)

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    def get_embedding(self, adjacency, embedding):

        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)

        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        embs = embedding
        item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), embs)
        return item_embeddings

    def intra_gate2(self, adjacency, trans1,embedding1, embedding2):
        # v_attention to get embedding of type, and then gate to get final type embedding
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        adjacency = torch.sparse_coo_tensor(i, v, torch.Size(shape), dtype=torch.float32)
        matrix = adjacency.to_dense()
        matrix = trans_to_cuda(matrix)
        tran_emb2 = trans1(embedding2)


        alpha = torch.mm(embedding1, torch.transpose(tran_emb2, 1, 0))
        alpha = torch.nn.Softmax(dim=1)(alpha)
        alpha = alpha * matrix
        sum_alpha_row = torch.sum(alpha, 1).unsqueeze(1).expand_as(alpha) + 1e-8
        alpha = alpha / sum_alpha_row

        type_embs = torch.mm(alpha, embedding2)
        item_embeddings = type_embs
        return item_embeddings

    def inter_gate3(self, w1, w11, w2, w3,  emb1, emb2, emb3):
        # 4 to 1
        all_emb = torch.cat([emb1, emb2, emb3], 1)

        gate1 = torch.tanh(w1(all_emb) + w2(emb2))
        gate2 = torch.tanh(w1(all_emb) + w3(emb3))
        h_embedings = emb1 + gate1 * emb2 + gate2 * emb3
        return h_embedings

    def forward(self, user_sample_a, user_sample_b, item_sample_a, item_sample_b):
        device = user_sample_a.device
        Eua = self.domainA_user_embedding(device)
        Eub = self.domainB_user_embedding(device)


        A_embedding = self.A_embedding.weight
        B_embedding = self.B_embedding.weight
        A_pri_emb = self.A_price_embedding.weight
        B_pri_emb = self.B_price_embedding.weight
        A_cate_emb = self.A_category_embedding.weight
        B_cate_emb = self.B_category_embedding.weight

        Eia = [A_embedding]
        Eib = [B_embedding]

        for i in range(self.single_layer_num):
            Mua = Eua[-1]
            Mub = Eub[-1]

            if i != 0:
                Mua = reduce(torch.mm, [self.conv_au.to(device), Mua])
                Mub = reduce(torch.mm, [self.conv_bu.to(device), Mub])

                Mua = self.ac_func(self.linears_a[i](Mua).detach())+ Mua.detach()
                Mub = self.ac_func(self.linears_b[i](Mub).detach()) + Mub.detach()

                A_embedding = self.ac_func(self.linears_a[i](A_embedding).detach()) + A_embedding.detach()
                B_embedding = self.ac_func(self.linears_b[i](B_embedding).detach()) + B_embedding.detach()

            if self.graph_type == 'heterogeneous':
                domainA_item_embedding = (self.inter_gate3(
                    self.w_v_1, self.w_v_11, self.w_v_2, self.w_v_3, A_embedding,
                    self.get_embedding(self.A_adjacency_vp, A_pri_emb),
                    self.get_embedding(self.A_adjacency_vc, A_cate_emb)) +
                    self.get_embedding(self.A_adjacency, A_embedding))

                domainB_item_embedding = (self.inter_gate3(
                    self.w_v_1, self.w_v_11, self.w_v_2, self.w_v_3, B_embedding,
                    self.get_embedding(self.B_adjacency_vp, B_pri_emb),
                    self.get_embedding(self.B_adjacency_vc, B_cate_emb)) +
                    self.get_embedding(self.B_adjacency, B_embedding))

                domainA_price_embeddings = (self.inter_gate3(
                    self.w_p_1, self.w_p_11, self.w_p_2, self.w_p_3, A_pri_emb,
                    self.intra_gate2(self.A_adjacency_pv, self.tran_pv,  A_pri_emb, A_embedding),
                    self.intra_gate2(self.A_adjacency_pc, self.tran_pc, A_pri_emb, A_cate_emb)) +
                    self.get_embedding(self.A_adjacency_pp, A_pri_emb))

                domainB_price_embeddings = (self.inter_gate3(
                    self.w_p_1, self.w_p_11, self.w_p_2, self.w_p_3, B_pri_emb,
                    self.intra_gate2(self.B_adjacency_pv,  self.tran_pv,  B_pri_emb, B_embedding),
                    self.intra_gate2(self.B_adjacency_pc,  self.tran_pc,  B_pri_emb, B_cate_emb)) +
                    self.get_embedding(self.B_adjacency_pp, B_pri_emb))

                domainA_category_embeddings = (self.inter_gate3(
                    self.w_c_1, self.w_c_11, self.w_c_2, self.w_c_3, A_cate_emb,
                    self.intra_gate2(self.A_adjacency_cp,  self.tran_cp,  A_cate_emb, A_pri_emb),
                    self.intra_gate2(self.A_adjacency_cv,  self.tran_cv,  A_cate_emb, A_embedding)) +
                    self.get_embedding(self.A_adjacency_cc, A_cate_emb))

                domainB_category_embeddings = (self.inter_gate3(
                    self.w_c_1, self.w_c_11, self.w_c_2, self.w_c_3, B_cate_emb,
                    self.intra_gate2(self.B_adjacency_cp, self.tran_cp, B_cate_emb, B_pri_emb),
                    self.intra_gate2(self.B_adjacency_cv,  self.tran_cv, B_cate_emb, B_embedding)) +
                    self.get_embedding(self.B_adjacency_cc, B_cate_emb))

                A_embedding = domainA_item_embedding
                A_pri_emb = domainA_price_embeddings
                A_cate_emb = domainA_category_embeddings

                B_embedding = domainB_item_embedding
                B_pri_emb = domainB_price_embeddings
                B_cate_emb = domainB_category_embeddings

            if self.graph_type == 'homogeneous':
                domainA_item_embedding =self.get_embedding(self.A_adjacency, A_embedding)
                domainB_item_embedding =self.get_embedding(self.B_adjacency, B_embedding)
                A_embedding = domainA_item_embedding
                B_embedding = domainB_item_embedding

            if self.model_structure == "inter-user" or self.model_structure == "normal":
                Mua_from_b = self.ac_func(torch.matmul(self.HyGCN_a_u.to(device), Mub))
                Mub_from_a = self.ac_func(torch.matmul(self.HyGCN_b_u.to(device), Mua))

                gate_a = torch.sigmoid(self.gate_a(Mua) + self.gate_b(Mua_from_b))
                Mua_inter = gate_a * Mua + (1 - gate_a) * Mua_from_b

                gate_b = torch.sigmoid(self.gate_b(Mub) + self.gate_a(Mub_from_a))
                Mub_inter = gate_b * Mub + (1 - gate_b) * Mub_from_a


                Eua.append(Mua_inter)
                Eub.append(Mub_inter)

            if self.model_structure == "inter-item" or self.model_structure == "normal":
                Mia_from_b = self.ac_func(torch.matmul(self.HyGCN_a_i.to(device), B_embedding))
                Mib_from_a = self.ac_func(torch.matmul(self.HyGCN_b_i.to(device), A_embedding))

                gate_A = torch.sigmoid(self.gate_a(A_embedding) + self.gate_b(Mia_from_b))
                A_embedding = gate_A * A_embedding + (1 - gate_A) * Mia_from_b

                gate_B = torch.sigmoid(self.gate_b(B_embedding) + self.gate_a(Mib_from_a))
                B_embedding = gate_B * B_embedding + (1 - gate_B) * Mib_from_a

                Eia.append(A_embedding)
                Eib.append(B_embedding)

            if self.model_structure == "single" or self.model_structure == "inter-user":
                Eia.append(A_embedding)
                Eib.append(B_embedding)

            if self.model_structure == "single" or self.model_structure == "inter-item":
                Eua.append(Mua)
                Eub.append(Mub)

        Pua = torch.cat(Eua, dim=1)
        Pub = torch.cat(Eub, dim=1)

        Pia = torch.cat(Eia, dim=1)
        Pib = torch.cat(Eib, dim=1)

        score_a = torch.cosine_similarity(Pua[user_sample_a.to(device)], Pia[item_sample_a.to(device)], dim=1)
        score_b = torch.cosine_similarity(Pub[user_sample_b.to(device)], Pib[item_sample_b.to(device)], dim=1)

        return torch.clamp(score_a, 0, 1), torch.clamp(score_b, 0, 1)

