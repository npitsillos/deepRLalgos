import numpy as np


class Normalizer(object):
    
    def __init__(
            self,
            size,
            eps=1e-8,
            default_clip_range=np.inf,
            mean=0,
            std=1,
    ):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range
        self.sum = np.zeros(self.size, np.float32)
        self.sumsq = np.zeros(self.size, np.float32)
        self.count = np.ones(1, np.float32)
        self.mean = mean + np.zeros(self.size, np.float32)
        self.std = std * np.ones(self.size, np.float32)
        self.synchronized = True


    def update(self, v):
        if v.ndim == 1:
            v = np.expand_dims(v, 0)
        assert v.ndim == 2
        assert v.shape[1] == self.size
        self.sum += v.sum(axis=0)
        self.sumsq += (np.square(v)).sum(axis=0)
        self.count[0] += v.shape[0]
        self.synchronized = False


    def normalize(self, v, clip_range=None):
        if not self.synchronized:
            self.synchronize()
        if clip_range is None:
            clip_range = self.default_clip_range
        mean, std = self.mean, self.std
        if v.ndim == 2:
            mean = mean.reshape(1, -1)
            std = std.reshape(1, -1)
        return np.clip((v - mean) / std, -clip_range, clip_range)


    def denormalize(self, v):
        if not self.synchronized:
            self.synchronize()
        mean, std = self.mean, self.std
        if v.ndim == 2:
            mean = mean.reshape(1, -1)
            std = std.reshape(1, -1)
        return mean + v * std


    def synchronize(self):
        self.mean[...] = self.sum / self.count[0]
        self.std[...] = np.sqrt(
            np.maximum(
                np.square(self.eps),
                self.sumsq / self.count[0] - np.square(self.mean)
            )
        )
        self.synchronized = True
