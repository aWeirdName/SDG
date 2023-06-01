import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
# from torch_geometric.nn import GCNConv, GATConv, EdgeConv
import math
import torch.nn.functional as F
import networkx as nx 
# from graph_layer import GraphLayer
from layer import GAT
from utils import *
from infoNCE import InfoNCE
from Knn_graph import knn_graph
def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1,batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

    return batch_edge_index.long()


class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num, inter_num = 512):
        super(OutLayer, self).__init__()

        modules = []

        for i in range(layer_num):
            # last layer, output shape:1
            if i == layer_num-1:
                modules.append(nn.Linear( in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear( layer_in_num, inter_num ))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0,2,1)
                out = mod(out)
                out = out.permute(0,2,1)
            else:
                out = mod(out)

        return out



class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=100):
        super(GNNLayer, self).__init__()


        self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)

        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, edge_index, embedding=None, node_num=0):

        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True)
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index
  
        out = self.bn(out)
        
        return self.relu(out)


class SDG(nn.Module):
    def __init__(self, batch, edge_index_sets, node_num, dim=64, out_layer_inter_dim=256, input_dim=10, out_layer_num=1, topk=20, device='cuda:1', beta=0.01, lam=0.01):

        super(GDN, self).__init__()

        self.batch_size = batch
        self.beta = beta
        self.lam = lam
        self.device = device

        embed_dim = dim
        self.node_num = node_num
        self.embed_dim = dim
        
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)

        self.lin = nn.Linear(input_dim, embed_dim)
        self.tran = nn.Linear(2*embed_dim, embed_dim)
        self.tran1 = nn.Linear(2*embed_dim, embed_dim)


        self.node_embeddings = nn.Parameter(torch.randn(node_num, embed_dim), requires_grad=True)

        self.learned_graph = None

        self.out_layer = OutLayer(dim, node_num, out_layer_num, inter_num = out_layer_inter_dim)

        self.dp = nn.Dropout(0.2)
        self.mask = diag_block_mask(batch, node_num, device)
        self.GAT = GAT(nfeat=input_dim, nout=dim, dropout=0, alpha=0.2)
        self.GAT1 = GAT(nfeat=input_dim, nout=dim, dropout=0, alpha=0.2)
        self.GAT2 = GAT(nfeat=dim*2, nout=dim, dropout=0, alpha=0.2, concat=False)

    def get_d_graph(self, emb, sigma = 1):
        emb = self.lin(emb)
        learned_graph = cosin_sim(emb)

        learned_graph = torch.stack([learned_graph, 1-learned_graph], dim=-1)
        adj = torch.nn.functional.gumbel_softmax(learned_graph, tau=10, hard=True)
        adj = adj[:, :, 0].clone().reshape(self.node_num * self.batch_size , -1)
        adj *= self.mask
        return adj
    

    def get_s_graph(self, emb):
        #cosin_dis
        learned_graph = cosin_sim(emb)
        #gumbel mat
        learned_graph = torch.stack([learned_graph, 1-learned_graph], dim=-1)
        adj = torch.nn.functional.gumbel_softmax(learned_graph, tau=10, hard=True)
        #add self loop
        adj = adj[:, :, 0].clone().reshape(self.node_num , -1)
        n = self.node_num 
        adj += torch.eye(n,n).to(self.device)
        adj = adj.bool().int().float()
        self.s_adj = adj
        #adj for batch
        adj = adj.repeat(self.batch_size, self.batch_size)
        adj *= self.mask
        
        return adj


    def forward(self, data):

        x = data.clone().detach()
        batch_num, node_num, all_feature = x.shape
        x = x.view(-1, all_feature).contiguous()
        
        #get graph----------------------------------------------
        S = self.node_embeddings       
        self.Static_graph = self.get_s_graph(S)
        D = x #
        self.Dynamic_graph = self.get_d_graph(D) #

        #embeding-----------------------------------------------
        emb_for_batch = self.node_embeddings.repeat(batch_num,1)
        self.Emb_S,self.a1 = self.GAT(x, self.Static_graph, emb_for_batch)
        self.Emb_D,self.a2 = self.GAT1(x, self.Dynamic_graph, emb_for_batch) #

        #mix up----------------------------------------------------
        self.emb = self.Emb_S + 0.01*self.Emb_D


        #forcasting------------------------------------------------
        out = self.emb.view(batch_num, node_num, -1)

        out = torch.mul(out, self.node_embeddings) #
        
        out = out.permute(0,2,1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0,2,1)

        out = self.dp(out)
        out = self.out_layer(out)
        out = out.view(-1, node_num)
        # loss = 0 #self.build_loss()
        return out

    def p_tsen(self):
        from tsne import Tsne
        Tsne(self.Emb_D,'D')
        Tsne(self.Emb_S,'S')
        Tsne(self.emb,'emb')

    def p_heatmap(self, name):
        df = pd.DataFrame(self.Static_graph[0:51,0:51].detach().cpu().numpy())
        df.to_csv(f'./hp_res/{name}graph_s.csv',index= False)
        df = pd.DataFrame(self.Dynamic_graph[0:51,0:51].detach().cpu().numpy())
        df.to_csv(f'./hp_res/{name}graph_d.csv',index= False)

    def p_netwkx(self, name):
        df = pd.DataFrame(self.a1[0:51,0:51].detach().cpu().numpy())
        df.to_csv(f'./res/{name}adj_a1.csv',index= False)
        df = pd.DataFrame(self.a2[0:51,0:51].detach().cpu().numpy())
        df.to_csv(f'./res/{name}adj_a2.csv',index= False)

    def build_loss(self):
        
        # infoNCE = InfoNCE()
        
        # loss = self.lam * infoNCE(self.Emb_S, self.Emb_D)
        # loss += self.beta * self.adptive_loss(self.learned_graph, 0.1)
       
        return 0
        