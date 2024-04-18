import torch
import torch.nn as nn
import numpy as np
#from double_attention.models.model.MemoryAttention import Attention
from model.MemoryAttention import Attention
class QuestionModule(nn.Module):
    def __init__(self,mode_arg):
        super(QuestionModule, self).__init__()
        self.questions=nn.ModuleList([ nn.LSTM(input_size=512,hidden_size=256,num_layers=1,
                                              batch_first=True,dropout=0.2,bidirectional=True,device=mode_arg.device) for i in range(5)])

        self.SAC=Attention(mode_arg)
        self.WAC = Attention(mode_arg)

    def forward(self,inputs):
        #attention
        batch_num,sen_num,word_num,embedding_size=inputs.size()
        word_ouput=[]
        question_out=[]
        #C=Variable(torch.zeros(self.hidden_size)).cuda()
        for q in self.questions:
            #every question
            for bid in range(batch_num):
                sentences=inputs[bid]
                propt, _ = q(sentences)
                #words level attention
                w_output,w_soft=self.WAC(sentences,propt)
                word_ouput.append(w_output)
                #sentences level attention
            s_output,s_soft=self.SAC(word_ouput,propt)
            word_ouput=[]
            question_out.append(s_output)

        return question_out