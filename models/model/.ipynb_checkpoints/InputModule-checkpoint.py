import torch
import torch.nn as nn
from transformers import BertModel,BertTokenizer

class InputModule(nn.Module):
    def __init__(self, mode_arg):
        super().__init__()
        self.device=mode_arg.device
        self.bert=BertModel.from_pretrained('bert-base-uncased',cache_dir='../utils/bert',output_hidden_states=True).to(mode_arg.device)
        self.tokenizer=BertTokenizer.from_pretrained('bert-base-uncased',cache_dir='../utils/bert')
        self.bilstm=nn.LSTM(input_size=mode_arg.BL_input,hidden_size=mode_arg.BL_hidden,
                            num_layers=mode_arg.BL_layers,bidirectional=True,batch_first=True).to(mode_arg.device)
        self.dropout=nn.Dropout(mode_arg.Bert_dropout).to(mode_arg.device)#0.4
        self.linear=nn.Linear(768,768)
    def forward(self, inputs):
        '''
        input（batch,sentence,word_id)
        '''
        tokenizer_output=[]
        bert_output=[]

        #填充<PAD>
        max_len=max([len(sentence)for batch in inputs for sentence in batch])+2

        for batch in inputs:
            tokenizer_output.append([self.tokenizer.encode_plus(sentence,padding='max_length',max_length=max_len,return_tensors='pt',) for sentence in batch])
        # for batch in token2ids :
        #     tokenizer_output.append([  self.tokenizer(sentence,padding="max_length",max_length=max_len,return_tensors='pt')for sentence in batch])

        #bert 输出
        b=[]
        s=[]

        for batch in tokenizer_output :
            for sentence in batch:
                sentence.to(self.device)
                #print(sentence['input_ids'].shape)
                with torch.no_grad():
                    output=self.bert(**sentence)
        #bert_output.append([  self.bert(**sentence)for sentence in batch])
        #[0]last_hidden          [1]pooler_output

                pooled=[]
                for i in range(2):
                    seq=output[2][-i]
                    pooled+=[torch.mean(seq,dim=1,keepdim=True)]
                pooled=torch.mean(torch.cat(pooled,dim=1),1)
                pooled=self.linear(pooled)
                s.append(torch.cat((pooled,output[0][0][1:-1]),0))#remove [cls][sep]
            b.append(s)
            s = []

        #b=(4,12,179,768)
        bi_output=[]
        for sentence in b:

            output,_=self.bilstm(torch.stack(sentence))
            output=self.dropout(output)
            bi_output.append(output)


        return torch.stack(bi_output)





