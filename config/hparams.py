import config.config as config


class Hparams:
    def __init__(self):
        self.log = config.parameter['log']
        self.logger = None
        self.test_size = config.parameter['test_size']
        self.cuda_index = config.parameter['cuda_index']
        self.optimizer = config.parameter['optimizer']
        self.loss_func = config.parameter['loss_func']
        self.metrics = config.parameter['metrics']
        self.learningrate = config.parameter['learningrate']
        self.device = None  #config.parameter['device']
        self.shuffle = config.parameter['shuffle']
        self.learningrate = config.parameter['learningrate']
        self.shuffle = config.parameter['shuffle']
        self.use_dnn = config.parameter['use_dnn']
        self.use_cin = config.parameter['use_cin']
        self.linear_layer_size = config.parameter['linear_layer_size']
        self.dnn_hidden_units= config.parameter['dnn_hidden_units']
        self.cin_layer_size= config.parameter['cin_layer_size']
        self.cin_split_half= config.parameter['cin_split_half']
        self.cin_activation= config.parameter['cin_activation']
        self.l2_reg_linear= config.parameter['l2_reg_linear']
        self.l2_reg_embedding= config.parameter['l2_reg_embedding']
        self.l2_reg_dnn= config.parameter['l2_reg_dnn']
        self.l2_reg_cin= config.parameter['l2_reg_cin']
        self.dnn_dropout= config.parameter['dnn_dropout']
        self.dnn_activation= config.parameter['dnn_activation']
        self.init_std= config.parameter['init_std']
        self.seed= config.parameter['seed']
        self.dnn_use_bn= config.parameter['dnn_use_bn']
        self.task= config.parameter['task']
        self.init_method= config.parameter['init_method']

    def values(self):
        return config.parameter


