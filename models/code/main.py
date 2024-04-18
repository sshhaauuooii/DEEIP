import time
import sys
import argparse
import torch
print(torch.cuda.is_available())
import random
import os
import numpy as np
import pickle
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("==========================>"+os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# from double_attention.models.config.mode_arg import mode_arg
# from double_attention.models.code.dataset import mydata
# from double_attention.models.model.DoubelAttention import DoubelAttention
from config.mode_arg  import mode_arg
from dataset import CFEEDdata,MUCdata
from model.DoubelAttention import DoubelAttention
from utils.evaluate import recover_label,get_macro_avg,CFEED_evaluation

def time_chek(otime,time,e,all_e):
    log(otime,f'===================================')
    log(otime,f'周期{e+1}已完成')
    log(otime,f'当前用时{time:0.2f}')
    log(otime,f'总进度{(e+1)} --> {all_e}')
    log(otime,f'===================================')

def log(time,string):
    with open(f'../../Muc34/log/{time:0.0f}','a') as f:
        f.writelines('\n'+string)
    print(string)


def evaluate(model,conf):
    if conf.dataset=='MUC':dataset=MUCdata(conf,mode='dev')

    if conf.dataset=='CFEED':dataset=CFEEDdata(conf,mode='dev')
    dataloder = DataLoader(dataset, conf.eval_batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    sequences=[]
    pred_results = []
    gold_results = []
    doc_ids=[]
    model.eval()
    for eval_id, (eval_data, eval_tag,doc_id) in enumerate(dataloder):
        pre = model.pre(eval_data,eval_tag)

        pred_label, gold_label = recover_label(pre, eval_tag)
        pred_results += pred_label
        gold_results += [gold_label]

        sequences+=eval_data
        doc_ids+=doc_id
    if conf.dataset=='MUC':p, r, f = get_macro_avg(sequences, pred_results, doc_ids,to_print=True)
    if conf.dataset=='CFEED':p, r, f=CFEED_evaluation(conf.eventype,sequences,pred_results,gold_results)
    return p, r, f


def test(otime,conf):
    log(otime,"Testing models...")

    model=DoubelAttention(conf)

    chekpoint = torch.load(conf.model_path,map_location='cpu')
    model.load_state_dict(chekpoint['state_dict'])
    model=model.to(conf.device)
    best_f1 = chekpoint['best_f1']
    #print(model)
    precision,recall,f1score=evaluate(model,conf)
    log(otime,
                f'precision：{precision}\nrecall：{recall}\nf1score：{f1score}')

def train(conf,otime):
    log(otime,"Training models...")
    model=DoubelAttention(conf)
    model=model.to(conf.device)
    #print(model)
    if conf.dataset=='MUC':train_dataset=MUCdata(conf,mode='train')
    if conf.dataset=='CFEED':train_dataset=CFEEDdata(conf,mode='train')

    train_dataloder=DataLoader(train_dataset,conf.train_batch_size,shuffle=False,collate_fn=train_dataset.collate_fn)

    best_f1=-1
    fist_time=time.time()
    start_epch=-1
    if conf.conti:
        chekpoint=torch.load(conf.model_path)
        model.load_state_dict(chekpoint['state_dict'])
        start_epch=chekpoint['epoch']
        best_f1=chekpoint['best_f1']
        
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for e in range(start_epch+1,conf.epoch):
        
        #opt = torch.optim.AdamW(model.parameters(), lr=lr)
        all_loss=0
        # train
        model.train()
        for index,(train_data,train_tag,doc_id) in enumerate(train_dataloder):
            train_loss=model.loss_function(train_data,train_tag)
            train_loss.backward()
            all_loss+=float(train_loss)
            opt.step()
            opt.zero_grad()
            if index%5==0:
                log(otime,str(index+1)+'/'+str(train_dataset.__len__()/conf.train_batch_size)+f'train loss --> {train_loss:0.2f}')
        if e%1==0:
            # eval
            precision,recall,f1score=evaluate(model,conf)
            log(otime,
                f'all_loss={all_loss}\nprecision:{precision}\nrecall:{recall}\nf1score:{f1score}')
            log(otime,f'best f1score==> {best_f1}')
            if f1score>best_f1:
                log(otime,f'save model..')
                best_f1=f1score
                dic={
                    'state_dict':model.state_dict(),
                    'epoch':e,
                    'best_f1':f1score
                }
                torch.save(dic,conf.model_path)
        time_chek(otime,time.time()-fist_time,e,conf.epoch)
    log(otime,f'time of already train is {time.time()-fist_time}.')

def seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def config(args):
    conf=mode_arg(args.dataset)

    conf.epoch=args.epoch
    conf.mode=args.mode
    conf.device=args.device
    #train
    #conf.f1score=args.f1score
    conf.conti=args.conti

    return conf

if __name__=='__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',help="",default='train',choices=['train','test','dev'])
    parser.add_argument('--epoch', help="", default=50)#
    parser.add_argument('--device', help="", default='cuda')
    parser.add_argument('--conti', help="True or False",default=False)#

    parser.add_argument('--dataset', help="", default='CFEED',choices=['CFEED','MUC'])

    conf=config(parser.parse_args())
    seed(111)
    print("MODE:train")

    otime = time.time()

    if conf.mode=='train':
        log(otime,str(conf.__dict__))
        if conf.dataset=='CFEED':
            for even_ in conf.CFEEDeventype:
                conf.role_num = conf.role2num[even_]
                conf.model_path = conf.model_path + even_ + "model.pth"
                train(conf,otime)
        else:train(conf,otime)

    if conf.mode=='test':
        log(otime,str(conf.__dict__))
        conf.eventype = even_
        conf.role_num = conf.role2num[even_]
        conf.model_path = conf.model_path + even_ + "model.pth"
        test(otime,conf)

