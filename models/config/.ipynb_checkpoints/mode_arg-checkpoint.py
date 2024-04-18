


class mode_arg():
    def __init__(self):

        self.mode='test'
        self.epoch=10
        self.device='cuda'


        #train
        self.train_batch_size=4
        self.eval_batch_size=1
        self.f1score=None
        self.model_path='../../Muc34/best_model/model.pth'
        self.conti=None
        self.logic_batch=8

        #model
        self.max_position_embeddings=512
        self.Bert_dropout=0.4


        self.BL_input=768
        self.BL_layers=1
        self.BL_hidden=256


        self.convs2d=False






