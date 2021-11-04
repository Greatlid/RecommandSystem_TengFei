#batch sampler
import numpy as np
import torch

class BatchSampler():
    def __init__(self, feature = None, label = None, shuffle=False, batch_size=256):
        self.feature = feature
        self.label = label
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.datalength = len(feature)


    def GetBatch(self):
        #产生一个batch的数据,注意label可能为none，需要进行判断
        if self.shuffle:
            index = torch.randperm(self.datalength)
        else:
            index = torch.arange(0, self.datalength)

        curcnt = 0
        curbatch_feature, curbatch_label= [], []
        for i, ind in enumerate(index):
            curbatch_feature.append(self.feature[ind])
            if self.label:curbatch_label.append(self.label[ind])
            curcnt += 1
            if curcnt >= self.batch_size or i+1 == self.datalength:
                yield curbatch_feature, curbatch_label
                curcnt = 0
                curbatch_feature, curbatch_label = [], []
