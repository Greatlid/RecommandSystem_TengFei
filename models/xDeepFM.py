import torch.nn as nn
from models.Base_Model import *
from tensorflow.python.keras.callbacks import History



class ExtremeDeepFMModel(BaseModel):
    def __init__(self, use_dnn = True, use_cin = True, linear_layer_size = (256, 1) ,dnn_hidden_units = (256, 256),
                 cin_layer_size = (256, 128,),  cin_split_half = True, cin_activation = 'relu',
                 l2_reg_linear = 0.00001, l2_reg_embedding = 0.00001, l2_reg_dnn = 0,
                 l2_reg_cin = 0, init_std = 0.0001, seed = 1024, dnn_dropout = 0,
                 dnn_activation = 'relu', dnn_use_bn = False, task = 'binary',
                 device = 'cpu', gpus = None):
        super(ExtremeDeepFMModel,self).__init__()
        self.use_dnn = use_dnn
        self.use_cin = use_cin
        self.gpus = gpus
        self.history = History()
        self.linear_model = nn.Linear(
            linear_layer_size[0], linear_layer_size[1])
        if self.use_dnn:
            pass
            # self.dnn = nn.Linear(
            # linear_layer_size[0], linear_layer_size[1])
            # self.add_regularization_weight(
            #     filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)

        if self.use_cin:
            pass
            # self.cin = nn.Linear(
            # linear_layer_size[0], linear_layer_size[1])
            # self.add_regularization_weight(filter(lambda x: 'weight' in x[0], self.cin.named_parameters()),
            #                                l2=l2_reg_cin)
        self.to(device)

    def forward(self, x):
        # process data
        feat_emb = self._build_embedding(x)  #usernum*fieldnum*emb_dim
        logit = self._build_linear(x)
        if self.use_cin:
            logit += self._build_extreme_FM(feat_emb)
        if self.use_dnn:
            logit += self._build_dnn(feat_emb)
        return logit

    def _build_linear(self, feat_linear):
        pass

    def _build_extreme_FM(self, feat_emb):
        pass

    def _build_dnn(self, feat_emb):
        pass

    def _build_embedding(self, feat_list):
        pass

