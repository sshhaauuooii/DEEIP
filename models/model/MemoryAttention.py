import torch
import torch.nn as nn
import numpy as np
from torch.autograd.variable import Variable
import torch.nn.init as init
import torch.nn.functional as F
import math

def position_encoding(embedded_sentence):
    '''
    embedded_sentence.size() -> (#batch, #sentence, #token, #embedding)
    l.size() -> (#sentence, #embedding)
    output.size() -> (#batch, #sentence, #embedding)
    '''
    _, _, slen, elen = embedded_sentence.size()

    l = [[(1 - s/(slen-1)) - (e/(elen-1)) * (1 - 2*s/(slen-1)) for e in range(elen)] for s in range(slen)]
    l = torch.FloatTensor(l)
    l = l.unsqueeze(0) # for #batch
    l = l.unsqueeze(1) # for #sen
    l = l.expand_as(embedded_sentence)
    weighted = embedded_sentence * Variable(l.cuda())
    return torch.sum(weighted, dim=2).squeeze(2) # sum with tokens


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1600):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model) # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term) # [max_len, d_model/2]
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)
    def forward(self, x):
        """
        :param x: [x_len, batch_size, emb_size]
        :return: [x_len, batch_size, emb_size]
        """
        x=x.transpose(1,0)
        x = x + self.pe[:x.size(0), :]  # [x_len, batch_size, d_model]
        return self.dropout(x.transpose(1,0))


class S_Attention(nn.Module):
    def __init__(self, mode_arg):
        super().__init__()
        self.D_A=D_Attention(mode_arg)

    def forward(self,inputs):
        #inputs(sentence,word,hidden)
        #propt,_=sq(inputs)
        output=self.D_A(inputs)

        return output

class Attention(nn.Module):
    def __init__(self, mode_arg):
        super().__init__()
        self.Attention=AttentionCell(mode_arg).to(mode_arg.device)
        self.ps=PositionalEncoding(768)
    def forward(self, inputs, qustions):
        # inputs[12,179,512]
        # inpuys[4,?,?]
        inputs=self.ps(inputs)
        qustions=self.ps(qustions)
        output=[]
        outsoft=[]
        for input,qustion in zip(inputs,qustions):
            out,soft=self.Attention(input,qustion)
            output.append(out)
            outsoft.append(soft)

        return torch.stack(output),outsoft



class AttentionCell(nn.Module):
    def __init__(self,mode_arg):
        super().__init__()

    def forward(self, inputs,qustions):
        #inputs[179,512]
        # qustions[179,512]
        k=qustions.transpose(1,0)
        q=v=inputs
        d_k=q.size(-1)
        scores=torch.matmul(q,k)/math.sqrt(d_k)

        p_attn=F.softmax(scores,dim=-1)

        return torch.matmul(p_attn,v),p_attn



class D_Attention(nn.Module):
    def __init__(self,mode_arg):
        super(D_Attention, self).__init__()
        self.device=mode_arg.device
        self.attention=Attention(mode_arg)
    def forward(self,inputs):

        k=inputs.transpose(0,1)
        q=v=inputs
        sentence, word, hidden = q.shape
        soft=self.matmul(q,k)
        row=[]
        soft_index = torch.sort(soft, dim=1, descending=False)
        for i in range(sentence):
            fusion_list = []
            soft_, index_ = soft_index[0][i], soft_index[1][i]

            if len(inputs)>2:
                for soft_one, index_one in zip(soft_[:3], index_[:3]):
                    fusion_list.append(v[index_one])
                out_cat = torch.cat((fusion_list[0], fusion_list[1],fusion_list[2]), dim=0)

            elif len(inputs)==2:
                for soft_one, index_one in zip(soft_[:2], index_[:2]):
                    fusion_list.append(v[index_one])
                out_cat = torch.cat((fusion_list[0], fusion_list[1]), dim=0)

            else:
                for soft_one, index_one in zip(soft_[:1], index_[:1]):
                    fusion_list.append(v[index_one])
                out_cat = fusion_list[0]

            fusion_, _ = self.attention(out_cat.unsqueeze(0),out_cat.unsqueeze(0))

            row.append(fusion_[0][:word-1])
        return torch.stack(row)

    def matmul(self,x:torch.FloatTensor,y:torch.FloatTensor):
        sentence,word,hidden=x.shape
        column=[]
        for i in range(sentence):
            row=torch.stack([self.matmulCell(x[i, 0, :],y[0, j, :]) for j in range(sentence)])
            soft=row.sum(dim=1)

            column.append(soft)
        return torch.stack(column)

    def matmulCell(self,x,y):

        return abs(x-y)


