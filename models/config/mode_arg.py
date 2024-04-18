


class mode_arg():
    def __init__(self,dataset):

        self.mode='test'
        self.epoch=10
        self.device='cuda'
        self.dataset=dataset

        #train
        self.train_batch_size=4#4
        self.eval_batch_size=1
        self.f1score=None
        if self.dataset=='MUC':
            self.model_path='../../Muc34/best_model/model.pth'
            self.pre_model='bert-base-uncased'
            self.role_num=5

            self.skip=0.30
            self.eventype=None
            self.role_num=None
            self.b = 0.0
            self.skip = 0.30
        if self.dataset=='CFEED':
            self.model_path='../../CFEED/best_model/'
            self.pre_model = f'bert-base-chinese'
            self.role2num={
                'FREEZE': 5,
                'OWUW': 3,
                'PLEDGE': 5,
            }
            self.eventype=None
            self.role_num=None

            self.skip = 0.50
        
        self.CFEEDeventype=['FREEZE', 'OWUW','PLEDGE']
        self.conti=None
        self.logic_batch=8

        #model
        self.max_position_embeddings=768
        self.Bert_dropout=0.4


        self.BL_input=768
        self.BL_layers=1
        self.BL_hidden=768







