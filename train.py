import torch
from models.xDeepFM import ExtremeDeepFMModel as xDeepFM
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error, accuracy_score

def LoadData(hparams):
    #load feature, label, feature: each user:[fieldid:featureid:value]  list.sparse tensor
    #load feature, each user:[[[[feature_id, value], ...], [...]],      [[feature_row], [feature_col], [value]]]
    #数据集文件以str形式存储，str eval, with open...
    with open(hparams.feature_file_path, 'r') as fp:
        feature = fp.read()
        feature = eval(feature)
    with open(hparams.label_file_path, 'r') as fp:
        label = fp.read()
        label = eval(label)
    return feature, label


def train(hparams):
    params = hparams.values()
    for key, val in params.items():
        hparams.logger.info(str(key) + ':' + str(val))

    feature, label = LoadData(hparams)
    if len(feature[0]) != hparams.field_num:
        hparams.field_num = len(feature[0])
    # feature, label = [[[[[1,1],[2,1]], [[3,1],[4,1],[5,0.1]]],     [[0,0,0,0,0],[1,2,3,4,5],[1,1,1,1,0.1]]],
    #                   [[[[1,1],[2,1]], [[3,1],[6,1],[5,0.5]]],     [[1,1,1,1,1],[1,2,3,5,6],[1,1,1,0.5,1]]]], [1,0]
    #split data for train and test
    X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=hparams.test_size, shuffle = hparams.shuffle,random_state=1234)

    #define model
    if hparams.cuda_index[0] >= 0 and torch.cuda.is_available():
        print('cuda ready...')
        hparams.device = 'cuda'
    else:
        print('cpu...')
        hparams.device = 'cpu'

    model = xDeepFM(field_num = hparams.field_num,feature_count = hparams.feature_count,feature_dim = hparams.feature_dim,
                    use_dnn = hparams.use_dnn, use_cin = hparams.use_cin,
                    linear_layer_size = (hparams.feature_count,1),
                    dnn_hidden_units = hparams.dnn_hidden_units, cin_layer_size = hparams.cin_layer_size,
                    cin_split_half = hparams.cin_split_half, cin_activation = hparams.cin_activation,
                    l2_reg_linear = hparams.l2_reg_linear, l2_reg_embedding = hparams.l2_reg_embedding,
                    l2_reg_dnn = hparams.l2_reg_dnn, l2_reg_cin = hparams.l2_reg_cin, init_std = hparams.init_std,
                    seed = hparams.seed, dnn_dropout = hparams.dnn_dropout, dnn_activation = hparams.dnn_activation,
                    dnn_use_bn = hparams.dnn_use_bn, task = hparams.task, device= hparams.device, gpus=hparams.cuda_index)
    model.compile(optimizer= hparams.optimizer,lr=hparams.learningrate, loss=hparams.loss_func, metrics=hparams.metrics)
    model._get_initializer(hparams.init_method)
    history = model.fit(X_train, y_train, initial_epoch=hparams.initial_epoch,batch_size=hparams.batch_size, epochs=hparams.epochs, shuffle=hparams.shuffle)

    #hparams.logger.info
    pred_ans = model.predict(X_test, hparams.batch_size)
    pred_target = np.array(y_test)
    mse = mean_squared_error(pred_target, pred_ans)
    acc = accuracy_score(pred_target, pred_ans)
    hparams.logger.info('mse:' + str(mse))
    hparams.logger.info('acc:' + str(acc))
    print(mse, acc)
