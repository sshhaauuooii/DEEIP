import json
import random
import argparse

from transformers import BertTokenizer
tags=["perp_individual_id","perp_organization_id","phys_tgt_id","hum_tgt_name","incident_instrument_id"]
tokenizer=BertTokenizer.from_pretrained('bert-base-uncased',cache_dir='../../models/utils/bert')

def create_text2tag(datas):
    texts=[]
    perp_individuals=[]
    perp_organizations=[]
    phys_tgts=[]
    hum_tgt_names=[]
    incident_instruments=[]

    for data in datas:
        doc=datas[data]['doc']
        roles=datas[data]['roles']
        if list(roles.values())==[[],[],[],[],[]]:
            continue
        texts.append(tag_generate(doc,roles))
    print("The examples' number:"+str(len(texts)))
    return texts

def tag_generate(docs,roles):
    individual = roles["perp_individual_id"]
    organization = roles["perp_organization_id"]
    phys_tgt = roles["phys_tgt_id"]
    hum_tgt = roles["hum_tgt_name"]
    instrument = roles["incident_instrument_id"]

    #的
    docs=docs.lower().replace('\n',' ').replace('\"',"")

    # #bert添加新词汇
    # r_docs=''
    # for char in docs:
    #     if (char.isalpha()==False)and (char!=' '):
    #         r_docs+=' '+char+' '
    #     else:r_docs+=char
    #
    tag_individual=[]
    tag_organization=[]
    tag_phys_tgt=[]
    tag_hum_tgt=[]
    tag_instrument=[]
    sentences=docs2sentences(docs)
    for sentence in sentences:

        tag_individual.append(tag_match(sentence,individual))
        tag_organization.append(tag_match(sentence,organization))
        tag_phys_tgt.append(tag_match(sentence,phys_tgt))
        tag_hum_tgt.append(tag_match(sentence,hum_tgt))
        tag_instrument.append(tag_match(sentence,instrument))

    return str2list(sentences,tag_individual,tag_organization,tag_phys_tgt,tag_hum_tgt,tag_instrument)


def str2list(docs,tag_individuals,tag_organizations,tag_phys_tgts,tag_hum_tgts,tag_instruments):
    d_doc=[]
    d_tag_individual=[]
    d_tag_organization=[]
    d_tag_phys_tgt=[]
    d_tag_hum_tgt=[]
    d_tag_instrument=[]
    dic = dict()
    for doc,tag_individual,tag_organization,tag_phys_tgt,tag_hum_tgt,tag_instrument in zip(docs,tag_individuals,tag_organizations,tag_phys_tgts,tag_hum_tgts,tag_instruments):
        assert len(doc)==len(tag_individual)==len(tag_instrument)==len(tag_organization)==len(tag_phys_tgt)==len(tag_hum_tgt)
        if len(doc)<2:
            continue
        r_doc = []
        r_tag_individual = []
        r_tag_organization = []
        r_tag_phys_tgt = []
        r_tag_hum_tgt = []
        r_tag_instrument = []

        #front_id = 1
        front_id = 0
        for id,char in enumerate(doc):
            if char==' ':
                if id==front_id:
                    front_id=id+1
                    continue
                r_doc.append( doc[front_id:id])
                r_tag_individual.append(tag_individual[front_id])

                r_tag_organization.append(tag_organization[front_id])

                r_tag_phys_tgt.append(tag_phys_tgt[front_id])

                r_tag_hum_tgt.append(tag_hum_tgt[front_id])

                r_tag_instrument.append(tag_instrument[front_id])

                front_id=id+1
            else:
                if char.isalpha()==False:
                    if id==front_id:
                        r_doc.append(char)
                        r_tag_individual.append(tag_individual[id])

                        r_tag_organization.append(tag_organization[id])

                        r_tag_phys_tgt.append(tag_phys_tgt[id])

                        r_tag_hum_tgt.append(tag_hum_tgt[id])

                        r_tag_instrument.append(tag_instrument[id])

                        front_id=id+1
                    else:
                        r_doc.append(doc[front_id:id])
                        # if len(tag_individual)==1:
                        #     continue
                        # print(len(tag_individual),front_id)
                        r_tag_individual.append((tag_individual[front_id]))
                        r_doc.append(char)
                        r_tag_individual.append(tag_individual[id])

                        r_tag_organization.append(tag_organization[front_id])

                        r_tag_phys_tgt.append(tag_phys_tgt[front_id])

                        r_tag_hum_tgt.append(tag_hum_tgt[front_id])

                        r_tag_instrument.append(tag_instrument[front_id])

                        r_tag_organization.append(tag_organization[id])

                        r_tag_phys_tgt.append(tag_phys_tgt[id])

                        r_tag_hum_tgt.append(tag_hum_tgt[id])

                        r_tag_instrument.append(tag_instrument[id])

                        front_id=id+1
        d_doc.append(r_doc)
        d_tag_individual.append(r_tag_individual)
        d_tag_organization.append(r_tag_organization)
        d_tag_phys_tgt.append(r_tag_phys_tgt)
        d_tag_hum_tgt.append(r_tag_hum_tgt)
        d_tag_instrument.append(r_tag_instrument)




    #添加到字典
    dic["perp_individual_id"]=d_tag_individual
    dic["perp_organization_id"]=d_tag_organization
    dic["phys_tgt_id"]=d_tag_phys_tgt
    dic["hum_tgt_name"]=d_tag_hum_tgt
    dic["incident_instrument_id"]=d_tag_instrument
    dic["sentences"]=d_doc
    return dic



def tag_match(docs,all_tags):
    r_tags=[]
    r_tags+=len(docs)*['O']
    if all_tags:
        for tags in all_tags:

            tags.sort(key=lambda x:len(x.split()))
            #paixu

            for tag in tags:
                tag=tag.lower().replace('[','(').replace(']',')')
                # if tag not in docs:
                #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                #     continue

                id=0
                l=len(docs)
                while(id>-1):
                    id=docs.find(tag,id,l)
                    if id>-1:
                        r_tags[id]='B'
                        r_tags[id+1:id+(len(tag))]=['I']*(len(tag)-1)
                        id=id+len(tag)

    assert len(docs)==len(r_tags)
    return r_tags

def docs2sentences(docs):
    sentences=[]
    sentence_flag=['.','!','?']


    flag=-1#无引号

    front_id=0
    for id,char in enumerate(docs):
        if char=='"':
            flag=-flag
            continue
        if char in sentence_flag:
            if flag>0:
                continue
            sentences.append(docs[front_id:id+1])
            front_id=id+1

    return sentences



if __name__=="__main__":

    A=argparse.ArgumentParser()
    A.add_argument('--method',choices=['train','dev','test'])
    args=A.parse_args()

    # if args.method=='test':
    #     with open("../"+args.method+".json","r")as f:
    #         datas=json.load(f)
    # else:
    #     with open("../"+args.method+"_full.json","r")as f:
    #         datas=json.load(f)
    with open("../"+ "test" + ".json", "r")as f:
        datas=json.load(f)


    examples=create_text2tag(datas)

    f=open("./pre_test.json","w")

    #f= open("./pre_"+args.method+'.json',"w")

    json.dump(examples,f)






