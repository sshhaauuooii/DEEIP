import time
import sys
import argparse
import torch
print(torch.cuda.is_available())
import random
import os
import numpy as np

from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("==========================>"+os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# from double_attention.models.config.mode_arg import mode_arg
# from double_attention.models.code.dataset import mydata
# from double_attention.models.model.DoubelAttention import DoubelAttention
from config.mode_arg  import mode_arg
from dataset import mydata
from model.DoubelAttention import DoubelAttention
from utils.evaluate import recover_label,get_macro_avg

def time_chek(otime,time,e,all_e):
    log(otime,f'===================================')
    log(otime,f'周期{e+1}已完成')
    log(otime,f'当前用时{time:0.2f}')
    log(otime,f'总进度{(e+1)} --> {all_e}')
    log(otime,f'===================================')

def log(time,string):
    with open(f'../../Muc34/datas/log/{time:0.0f}','a') as f:
        f.writelines('\n'+string)
    print(string)


def evaluate(model,mode,conf):
    dataset = mydata(conf, mode=mode)
    dataloder = DataLoader(dataset, conf.eval_batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    sequences=[]
    pred_results = []
    gold_results = []
    doc_ids=[]
    model.eval()
    for eval_id, (eval_data, eval_tag,doc_id) in enumerate(dataloder):
        # if len(eval_data)==0:
        #     continue
        #if eval_id>10:continue

        pre = model.pre(eval_data,eval_tag)

        pred_label, gold_label = recover_label(pre, eval_tag)
        pred_results += pred_label
        gold_results += gold_label
        sequences+=eval_data
        doc_ids+=doc_id
    p, r, f = get_macro_avg(sequences, pred_results, doc_ids,to_print=False)

    # for q_pre,q_tag in zip(all_pre,all_tag):
    #     r=macro_f1(q_pre,q_tag)
    #     recall.append(r[0])
    #     precision.append(r[1])
    #     f1.append(r[2])
    return p, r, f


def test(otime,conf):
    log(otime,"Testing models...")

    model=DoubelAttention(conf)

    chekpoint = torch.load(conf.model_path,map_location='cpu')
    model.load_state_dict(chekpoint['state_dict'])
    model=model.to(conf.device)
    best_f1 = chekpoint['best_f1']
    #print(model)
    precision,recall,f1score=evaluate(model,'test',conf)
    log(otime,
                f'{recall}\n{precision}\n{f1score}')
    log(otime,f'{recall.sum()/len(recall)}')
    log(otime, f'{precision.sum() / len(precision)}')
    log(otime, f'{f1score.sum() / len(f1score)}')

def train(conf,otime):
    log(otime,"Training models...")
    model=DoubelAttention(conf)
    model=model.to(conf.device)
    #print(model)
    train_dataset=mydata(conf,mode='train')
    train_dataloder=DataLoader(train_dataset,conf.train_batch_size,shuffle=False,collate_fn=train_dataset.collate_fn)

    #lr=[0.1,0.1,0.1,0.1,0.05,0.04,0.02,0.01,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005]

    best_f1=0
    fist_time=time.time()
    start_epch=-1
    if conf.conti:
        chekpoint=torch.load(conf.model_path)
        model.load_state_dict(chekpoint['state_dict'])
        start_epch=chekpoint['epoch']
        best_f1=chekpoint['best_f1']
        
    # bert_layer = list(map(id, model.inputM.bert.parameters()))
    # crf_layer=list(map(id, model.answerM.crf.parameters()))
    # base_params = filter(lambda p: id(p) not in bert_layer+crf_layer,model.parameters())
    lr=0.001
    # opt = torch.optim.AdamW([{'params': base_params},
    #                         {'params': model.inputM.bert.parameters(), 'lr': lr * 0.001},
    #                         {'params': model.answerM.crf.parameters(), 'lr': lr * 10.}], lr=lr)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for e in range(start_epch+1,conf.epoch):
        
        #opt = torch.optim.AdamW(model.parameters(), lr=lr)
        
        # train
        model.train()
        for index,(train_data,train_tag,doc_id) in enumerate(train_dataloder):
            if index>169:continue
            #if len(train_data)==0:continue
            train_loss=model.loss_function(train_data,train_tag)
            train_loss.backward()
            opt.step()
            opt.zero_grad()
            log(otime,str(index+1)+'/'+str(train_dataset.__len__()/conf.train_batch_size)+f'train loss --> {train_loss:0.2f}')
        if e%1==0:
            # eval
            precision,recall,f1score=evaluate(model,'test',conf)
            log(otime,
                f'{precision}\n{recall}\n{f1score}')
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
    conf=mode_arg()

    conf.epoch=args.epoch
    conf.mode=args.mode
    conf.device=args.device
    #train
    conf.f1score=args.f1score
    conf.conti=args.conti
    #model
    conf.max_position_embeddings=args.max_position_embeddings
    conf.BL_hidden=args.BL_hidden
    conf.BL_input=args.BL_input
    conf.BL_layers=args.BL_layers
    
    return conf


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',help="",default='test')
    parser.add_argument('--epoch', help="", default=30)#
    parser.add_argument('--device', help="", default='cuda')
    parser.add_argument('--conti', help="True or False",default=False)#
    #train
    parser.add_argument('--batch_size', help="", default=20)
    parser.add_argument('--f1score', help="", default='macro_f1')
    #model
    parser.add_argument('--max_position_embeddings', help="", default=512)
    parser.add_argument('--BL_input', help="", default=768)
    parser.add_argument('--BL_layers', help="", default=1)
    parser.add_argument('--BL_hidden', help="", default=256)



    conf=config(parser.parse_args())

    seed(111)
    print("MODE:train")

    otime = time.time()

    if conf.mode=='train':

        log(otime,str(conf.__dict__))
        train(conf,otime)
    if conf.mode=='test':
        log(otime,str(conf.__dict__))
        test(otime,conf)



