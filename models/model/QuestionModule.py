import torch
import torch.nn as nn
import numpy as np
#from double_attention.models.model.MemoryAttention import Attention
from model.MemoryAttention import Attention,S_Attention
class QuestionModule(nn.Module):
    def __init__(self,mode_arg):
        super(QuestionModule, self).__init__()
        self.W_questions=nn.ModuleList([ nn.LSTM(input_size=768,hidden_size=768,num_layers=1,
                                              batch_first=True,bidirectional=True,device=mode_arg.device) for i in range(mode_arg.role_num)])
        # self.S_questions = nn.ModuleList([nn.LSTM(input_size=768, hidden_size=768, num_layers=1,
        #                                         batch_first=True, bidirectional=True, device=mode_arg.device) for i in
        #                                 range(0)])
        self.linear=nn.Linear(1536,768)
        self.dropout=nn.Dropout(0.2)
        self.SAC=S_Attention(mode_arg)
        self.WAC = Attention(mode_arg)

    def forward(self,inputs):
        #attention
        batch_num=len(inputs)
        batch_ouput=[]
        question_out=[]
        #C=Variable(torch.zeros(self.hidden_size)).cuda()
        for wq in self.W_questions:
            #every question
            for bid in range(batch_num):
                sentences=inputs[bid]
                propt, _ = wq(sentences)
                #words level attention
                w_output,w_soft=self.WAC(sentences,self.linear(propt))
                #sentences level attention
                w_output=self.dropout(w_output)
                s_output=self.SAC(w_output)
                batch_ouput.append(s_output)
            question_out.append(batch_ouput)
            batch_ouput = []

        return question_out