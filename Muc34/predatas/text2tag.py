import json
import random
import argparse
from transformers import BertTokenizer
tags=["perp_individual_id","perp_organization_id","phys_tgt_id","hum_tgt_name","incident_instrument_id"]
tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')

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

def tag_generate(doc,roles):
    individual = roles["perp_individual_id"]
    organization = roles["perp_organization_id"]
    phys_tgt = roles["phys_tgt_id"]
    hum_tgt = roles["hum_tgt_name"]
    instrument = roles["incident_instrument_id"]
    #的
    doc=doc.lower()
    r_doc=''
    for char in doc:
        if (char.isalpha()==False)and (char!=' '):
            r_doc+=' '+char+' '
        else:r_doc+=char

    words=r_doc.strip().split()

    texts=[]
    tag_individuals=[]
    tag_organizations = []
    tag_phys_tgts = []
    tag_hum_tgts = []
    tag_instruments = []

    for word in words:
        wordspiece,tag_individual=word_tag(word,individual)
        _,tag_organization = word_tag(word, organization)
        _,tag_phys_tgt = word_tag(word, phys_tgt)
        _,tag_hum_tgt = word_tag(word, hum_tgt)
        _,tag_instrument = word_tag(word, instrument)

        texts=texts+[[i] for i in wordspiece]
        tag_individuals.extend(tag_individual)
        tag_organizations.extend(tag_organization)
        tag_phys_tgts.extend(tag_phys_tgt)
        tag_hum_tgts.extend(tag_hum_tgt)
        tag_instruments.extend(tag_instrument)

    return [texts,tag_individuals,tag_organizations,tag_phys_tgts,tag_hum_tgts,tag_instruments]

def word_tag(word,tags):
    r_tag=[]

    wordspiece=tokenizer.tokenize(word)
    l=len(wordspiece)
    if tags:
        for tag in tags[0]:
            tag=tag.lower()
            if word in tag.split():
                if word == tag.split()[0]:
                    r_tag=r_tag+['B']+(l-1)*['I']
                    break
                else:
                    r_tag = r_tag + ["I"] * (l)
                    break


    if r_tag==[]:
        r_tag=r_tag+['O']*l


    assert len(r_tag)==l
    return wordspiece,r_tag



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

    f=open("./"+ "test", "w")
    # if args.method=='test':
    #     f= open("./"+args.method,"w")
    # else:
    #     f= open("./"+args.method,"w")

    for text,tag_individual,tag_organization,tag_phys_tgt,tag_hum_tgt,tag_instrument in examples:

        for a,b,c,d,e,r in zip(text,tag_individual,tag_organization,tag_phys_tgt,tag_hum_tgt,tag_instrument):
            f.write(a+" "+b+" "+c+" "+d+" "+e+" "+r+"\n")

        f.write("#"*20+"\n")

















