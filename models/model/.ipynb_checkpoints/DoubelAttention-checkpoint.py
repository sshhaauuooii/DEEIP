import torch
import torch.nn as nn
import numpy as np

from .InputModule import InputModule

from .AnswerModule import AnswerModule
from .QuestionModule import QuestionModule
class DoubelAttention(nn.Module):
    def __init__(self,mode_arg):
        super().__init__()
        self.inputM=InputModule(mode_arg).to(mode_arg.device)
        #self.memoryM=Memory(mode_arg)
        self.questionM=QuestionModule(mode_arg).to(mode_arg.device)
        self.answerM=AnswerModule(mode_arg).to(mode_arg.device)


    def forward(self,input,tags):
        op_inputM=self.inputM(input)

        op_qustionM=self.questionM(op_inputM)


        op_answer=self.answerM(op_qustionM,tags)
        return op_answer


    def loss_function(self,inputs,tags):

        output=self.forward(inputs,tags)
        return output

    def pre(self,inpust,labels):
        op_inputM=self.inputM(inpust)
        op_qustionM=self.questionM(op_inputM)
        #input[5,4,12,179,512]
        pre = []
        b_pre=[]
        b_label=[]
        for input,label in zip(op_qustionM,labels):
            for batch_input,batch_label in zip(input,label):
                _1,mask_ids,label_id,_2=self.answerM.get_crfmask(batch_input,batch_label,skip=False)
                classifier=self.answerM.clssifier(batch_input)
                for sentence in batch_label:
                    b_label.append([i.item() for i in sentence])
                b_pre.extend([self.answerM.crf.decode(classifier,mask_ids),b_label])
                b_label=[]
            pre.append(b_pre)
            b_pre=[]
        return pre


