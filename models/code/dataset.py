import torch
import torch.nn as nn
import numpy as np
#from transformers import BertTokenizer
from torch.utils.data import Dataset
import json



class MUCdata(Dataset):
    def __init__(self,mode_arg,mode):
        super(MUCdata, self).__init__()
        self.tag2id=dict()
        self.tag2id['B']=1
        self.tag2id['I']=2
        self.tag2id["O"]=0
        self.max_position_embeddings=mode_arg.max_position_embeddings
        self.mode=mode_arg.mode
        self.device=mode_arg.device
        self.datas=self.get_datas(mode_arg,mode)
        #self.tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')


    def __getitem__(self, item):
        data=list(self.datas.values())[item]
        perp_individual_id=data['perp_individual_tags']
        perp_organization_id = data['perp_organization_tags']
        phys_tgt_id = data['phys_tgt_tags']
        hum_tgt_name = data['hum_tgt_tags']
        incident_instrument_id = data['incident_instrument_tags']
        doc=data['sentences']

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
            #assert len(self.tokenizer.encode(a)) <= self.max_position_embeddings

            r_doc.append(a)




        return r_doc,data['id'],r_perp_individual_id,r_perp_organization_id,  r_phys_tgt_id,  r_hum_tgt_name, r_incident_instrument_id


    def __len__(self):

        return len(self.datas)


    def get_datas(self,arg,mode:str)->json:


        with open('../../'+arg.dataset+'/datas/pre_'+mode+'.json','r') as f:
            datas=json.load(f)

        return datas



    def collate_fn(self,datas):
        max_list=0
        for data,doc_id,b,c,d,e,f in datas:
            if max_list<len(data):
                max_list=len(data)


        rdata=[]
        r_id=[]
        rb=[]
        rc=[]
        rd=[]
        re=[]
        rf=[]
        for data,doc_id,b,c,d,e,f in datas:
            #data=data+['[CLS]  [SEP]']*(max_list-len(data))
            #if len(data)<3:continue
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
            r_id.append(doc_id)
        return rdata,(rb,rc,rd,re,rf),r_id


class CFEEDdata(Dataset):
    def __init__(self,mode_arg,mode):
        super(CFEEDdata, self).__init__()
        self.tag2id=dict()
        self.tag2id['B']=1
        self.tag2id['I']=2
        self.tag2id["O"]=0
        self.max_position_embeddings=mode_arg.max_position_embeddings
        self.mode=mode_arg.mode
        self.device=mode_arg.device
        self.eventype=mode_arg.eventype
        self.datas=self.get_datas(mode_arg,mode)
        #self.tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')


    def __getitem__(self, item):
        data = list(self.datas[self.eventype])[item]
        role=[]
        for key in data.keys():
            if key=='text':continue
            role.append([torch.tensor([self.tag2id[i] for i in value] ,dtype=torch.long,device=self.device)for value in data[key]])

        # if data['text']==[]:
        #     print()
        return data['text'],'none',role


    def __len__(self):

        return len(self.datas[self.eventype])


    def get_datas(self,arg,mode:str)->json:


        with open('../../'+arg.dataset+'/datas/pre_'+mode+'.json','r',encoding='utf-8') as f:
            datas=json.load(f)

        return datas



    def collate_fn(self,datas):


        return [data[0] for data in datas],\
               [[datas[batchsize_index][2][role_index] for batchsize_index in range(len(datas))] for role_index in range(len(datas[0][2]))]       ,\
                [data[1] for data in datas]





