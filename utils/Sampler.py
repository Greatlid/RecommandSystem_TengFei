#batch sampler

class BatchSampler():
    def __init__(self, feature = None, label = None, shuffle=False, batch_size=256):
        self.feature = feature
        self.label = label
        self.shuffle = shuffle
        self.batch_size = batch_size


    def GetBatch(self):
        #产生一个batch的数据,注意label可能为none，需要进行判断
        yield self.feature, self.label