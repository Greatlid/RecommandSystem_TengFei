import torch.nn as nn
import torch

from models.Base_Model import *
from tensorflow.python.keras.callbacks import History
from models.CIN_Model import CIN
from models.DNN_Model import DNN
from models.model_utils import *

class ExtremeDeepFMModel(BaseModel):
    def __init__(self, field_num, feature_count,feature_dim, use_dnn = True, use_cin = True, linear_layer_size = (256, 1) ,dnn_hidden_units = (256, 256),
                 cin_layer_size = (256, 128,),  cin_split_half = True, cin_activation = 'relu',
                 l2_reg_linear = 0.00001, l2_reg_embedding = 0.00001, l2_reg_dnn = 0,
                 l2_reg_cin = 0, init_std = 0.0001, seed = 1024, dnn_dropout = 0,
                 dnn_activation = 'relu', dnn_use_bn = False, task = 'binary',
                 device = 'cpu', gpus = None):
        super(ExtremeDeepFMModel,self).__init__()
        self.feature_dim = feature_dim
        self.feature_count = feature_count
        self.use_dnn = use_dnn
        self.use_cin = use_cin
        self.gpus = gpus
        self.history = History()
        self.linear_model = nn.Linear(
            linear_layer_size[0], linear_layer_size[1])
        self.embeddings = nn.Embedding(self.feature_count, self.feature_dim)
        if self.use_dnn:
            self.dnn = DNN(field_num*feature_dim, dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)


        if self.use_cin:
            self.cin = CIN(field_num, cin_layer_size,
                           cin_activation, cin_split_half, l2_reg_cin, seed, device=device)
            self.add_regularization_weight(filter(lambda x: 'weight' in x[0], self.cin.named_parameters()),
                                           l2=l2_reg_cin)
        self.out = PredictionLayer(task, )
        self.to(device)


    def forward(self, x):
        # process data
        feat_emb = self._build_embedding(x)  #usernum*fieldnum*emb_dim
        logit = self._build_linear(x)
        if self.use_cin:
            logit += self._build_extreme_FM(feat_emb)
        if self.use_dnn:
            logit += self._build_dnn(feat_emb)
        y_pred = self.out(logit)
        return y_pred

    def _build_embedding(self, feat_list):
        user_emb = []
        for feature_singleuser in feat_list:
            feature_emb = feature_singleuser[0]
            field_emb = []
            for field in feature_emb:
                field_feature = torch.Tensor(field)
                if field_feature.dim() == 1:
                    field_feature = field_feature.unsqueeze(dim=0)
                field_index = field_feature[:,0].long()
                field_value = field_feature[:,1].unsqueeze(dim=0)
                emb_feature = self.embeddings(field_index)
                emb_field = torch.mm(field_value, emb_feature)
                field_emb.append(emb_field)
            user_emb.append(torch.cat(field_emb).unsqueeze(dim=0))
        user_emb = torch.cat(user_emb)
        return user_emb


    def _build_linear(self, feat_linear):
        row_index, col_index, value = [], [], []
        for feature_single in feat_linear:
            row_index += feature_single[1][0]
            col_index += feature_single[1][1]
            value += feature_single[1][2]
        fea_matrix = torch.sparse_coo_tensor(torch.LongTensor([row_index,col_index]), torch.Tensor(value),
                                    (len(feat_linear), self.feature_count))
        res_linear = self.linear_model(fea_matrix)
        return res_linear

    def _build_extreme_FM(self, feat_emb):
        res = self.cin(feat_emb)
        return res

    def _build_dnn(self, feat_emb):
        feat = feat_emb.reshape(feat_emb.shape[0],-1)
        res = self.dnn(feat)
        return res



