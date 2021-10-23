#a dictionary, define parameter

parameter = {
#'device': 'cuda',  # cpu or 'cuda'
'cuda_index': [0],  #cpu:-1  gpu: gpu index
'feature_file_path': None,
'label_file_path': None,
'log':'log',
'test_size': 0.5,
'optimizer':"adam",  #"sgd" "adagrad" "rmsprop"
'loss_func':"binary_crossentropy", #"mse" "mae"
'metrics':["binary_crossentropy", "auc", "mse", "accuracy"],
'learningrate':0.0001,
'shuffle':True,
'use_dnn': True,
'use_cin': True,
'linear_layer_size':(256, 1),
'dnn_hidden_units': (256, 256),
'cin_layer_size': (256, 128,),
'cin_split_half':  True,
'cin_activation': 'relu',  # 'sigmoid' 'linear' 'relu' 'dice' 'prelu'
'l2_reg_linear': 0.00001,
'l2_reg_embedding': 0.00001,
'l2_reg_dnn': 0,
'l2_reg_cin': 0,
'dnn_dropout': 0,
'dnn_activation' : 'relu', # 'sigmoid' 'linear' 'relu' 'dice' 'prelu'
'init_std': 0.0001,
'seed': 1024,
'dnn_use_bn': False,
'task': 'binary',
'init_method': 'xavier'
}
