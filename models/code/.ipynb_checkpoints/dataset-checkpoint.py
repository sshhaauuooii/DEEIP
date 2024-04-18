import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import Dataset
import json



class mydata(Dataset):
    def __init__(self,mode_arg,mode):
        super(mydata, self).__init__()
        self.tag2id=dict()
        self.tag2id['B']=1
        self.tag2id['I']=2
        self.tag2id["O"]=0
        self.max_position_embeddings=mode_arg.max_position_embeddings
        self.mode=mode_arg.mode
        self.device=mode_arg.device
        self.datas=self.get_datas(mode)
        self.tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')


    def __getitem__(self, item):

        perp_individual_id=self.datas[item]['perp_individual_tags']
        perp_organization_id = self.datas[item]['perp_organization_tags']
        phys_tgt_id = self.datas[item]['phys_tgt_tags']
        hum_tgt_name = self.datas[item]['hum_tgt_tags']
        incident_instrument_id = self.datas[item]['incident_instrument_tags']
        doc=self.datas[item]['sentences']

        r_doc=[]
        r_perp_individual_id=[]
        r_perp_organization_id=[]
        r_phys_tgt_id=[]
        r_hum_tgt_name=[]
        r_incident_instrument_id=[]
        for a,b,c,d,e,f in zip(doc,perp_individual_id,perp_organization_id,phys_tgt_id,hum_tgt_name,incident_instrument_id):
            #r_perp_individual_id.append(torch.tensor([ self.tag2id[i]for i in b],dtype=torch.long,device=self.device))
            r_perp_individual_id.append(torch.tensor([self.tag2id[i] for i in b] ,dtype=torch.long,device=self.device))
            r_perp_organization_id.append(torch.tensor([self.tag2id[i] for i in c] ,dtype=torch.long,device=self.device))
            r_phys_tgt_id.append(torch.tensor([self.tag2id[i] for i in d] ,dtype=torch.long,device=self.device))
            r_hum_tgt_name.append(torch.tensor([self.tag2id[i] for i in e] ,dtype=torch.long,device=self.device))
            r_incident_instrument_id.append(torch.tensor([self.tag2id[i] for i in f] ,dtype=torch.long,device=self.device))
            assert len(self.tokenizer.encode(a)) <= self.max_position_embeddings

            r_doc.append(a)

        return r_doc,r_perp_individual_id,r_perp_organization_id,  r_phys_tgt_id,  r_hum_tgt_name, r_incident_instrument_id


    def __len__(self):
        return len(self.datas)


    def get_datas(self,mode:str)->json:

        if mode=='test':
            with open('../../Muc34/datas/pre_test.json','r') as f:
                datas=json.load(f)
        if mode=='dev':
            with open('../../Muc34/datas/pre_dev.json','r') as f:
                datas=json.load(f)
        if mode=='train':
            with open('../../Muc34/datas/pre_train.json','r') as f:
                datas=json.load(f)


        return datas



    def collate_fn(self,datas):
        max_list=0
        for data,b,c,d,e,f in datas:
            if max_list<len(data):
                max_list=len(data)


        rdata=[]
        rb=[]
        rc=[]
        rd=[]
        re=[]
        rf=[]
        for data,b,c,d,e,f in datas:
            #data=data+['[CLS]  [SEP]']*(max_list-len(data))
            if len(data)<3:continue
            if max_list-len(b)>0:
                #num=max_list-len(b)
                for i in range(max_list-len(b)):
                    data.append(['#'])
                    b.append(torch.tensor([0],dtype=torch.long,device=self.device))
                    c.append(torch.tensor([0],dtype=torch.long,device=self.device))
                    d.append(torch.tensor([0],dtype=torch.long,device=self.device))
                    e.append(torch.tensor([0],dtype=torch.long,device=self.device))
                    f.append(torch.tensor([0],dtype=torch.long,device=self.device))
            rb.append(b)
            rc.append(c)
            rd.append(d)
            re.append(e)
            rf.append(f)
            rdata.append(data)
        return rdata,(rb,rc,rd,re,rf)





