import torch
import torch.nn as nn
import numpy as np
#from double_attention.models.model.crf import CRF
from model.crf import CRF
import random
class AnswerModule(nn.Module):
    def __init__(self,mode_arg):
        super(AnswerModule, self).__init__()
        self.device=mode_arg.device
        self.crf=CRF(3,batch_first=True).to(mode_arg.device)
        self.dropout=nn.Dropout(0.3).to(mode_arg.device)
        self.clssifier=nn.Sequential(
            nn.Linear(512, 256).to(mode_arg.device),
            nn.GELU(),
            nn.Linear(256, 3).to(mode_arg.device),
        )
    def forward(self,inputs,labels):
        #input[5,4,12,179,512]
        count=0
        loss = torch.tensor(0,dtype=torch.float,device=self.device,requires_grad=True)
        for input,label in zip(inputs,labels):
            for batch_input,batch_label in zip(input,label):
                r_batch_input,mask_ids,label_id,flag=self.get_crfmask(batch_input,batch_label)
                if flag:continue
                classifier=self.clssifier(r_batch_input)
                #loss.requires_grad=False
                loss=loss-self.crf(emissions=classifier,tags=label_id,mask=mask_ids,reduction='sum')
                count+=classifier.shape[0]
        return loss/count

    def get_crfmask(self,batch_input,batch_label,skip=True):
        """
        batch_q=[12,179,512]
        batch_label=[12,?]
        """
        crfmask=[]
        crflabel=[]
        flag = 0
        #0表示不跳过
        r_batch_input=[]

        for sentence,label in zip(batch_input,batch_label):

            sentence_len = sentence.size(0)
            label_len = len(label)

            if 1 not in label and 2 not in label and skip and (random.random()>0.30):continue


            r_batch_input.append(sentence)
            crfmask.append([1  if i<label_len else 0   for i in range(sentence_len) ])
            crflabel.append([label[i] if i<label_len   else torch.tensor(0,dtype=torch.long,device=self.device)  for i in range(sentence_len)])
        if r_batch_input:
            return torch.stack(r_batch_input), torch.ByteTensor(crfmask).to(self.device), torch.LongTensor(
                crflabel), flag
        flag=1
        return r_batch_input,torch.ByteTensor(crfmask).to(self.device),torch.LongTensor(crflabel),flag









