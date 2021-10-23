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

    def values(self):
        return config.parameter


