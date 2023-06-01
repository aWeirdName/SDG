import torch
import torch.nn as nn
import torch.nn.functional as F



class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        if concat==True:
            n = out_features
        else: 
            n = 0
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*(out_features+n), 1))) # 2*(out_fea+ node_emb_dim)  &  out_fea=node_dim
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj, emb):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh, emb)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # #--------------
        # import pandas as pd
        # df = pd.DataFrame(attention[0:51,0:51].detach().cpu().numpy())
        # df.to_csv(f'./res/adj.csv',index= False)
        # #--------------


        h_prime = torch.matmul(attention, Wh)
  
        if self.concat:
            return h_prime,attention #F.elu(h_prime)
        else:
            return h_prime,attention

    def _prepare_attentional_mechanism_input(self, Wh, emb=None):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        if emb!=None:
            WH_E = torch.cat((Wh, emb), dim=-1)
            dim = Wh.shape[1] + emb.shape[1]
        else:
            WH_E = Wh
            dim = Wh.shape[1]
        Wh1 = torch.matmul(WH_E, self.a[:dim, :])
        Wh2 = torch.matmul(WH_E, self.a[dim:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nout, dropout, alpha, concat = True):
        """one heads GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        # self.nhead = nheads
        # self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nfeat, nout, dropout=dropout, alpha=alpha, concat=concat)
        self.concat = concat
        self.bn = nn.BatchNorm1d(nout)
        self.relu = nn.ReLU()

    def forward(self, x, adj, emb=None):

        out,attention = self.out_att(x, adj, emb)
        if self.concat==False:
            out = self.bn(out)
            out = self.relu(out)

        # x = F.log_softmax(x, dim=1)
        return out,attention

