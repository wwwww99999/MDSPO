import numpy as np

class Normalize:
    def __init__(self, clip=1e6):
        self.x = 0
        self.mean = 0
        self.sumsq = 0
        self.var = 0
        self.std = 0
        self.count = 0
        self.clip = clip

    def push(self, x):
        self.x = x
        self.count += 1
        if self.count == 1:
            self.mean = x
        else:
            old_mean = self.mean.copy()
            self.mean += (x - self.mean) / self.count
            self.sumsq += (x - old_mean) * (x - self.mean)
            self.var = self.sumsq / (self.count - 1)
            self.std = np.sqrt(self.var)
            self.std = np.maximum(self.std, 1e-2)

    def get_mean(self):
        return self.mean

    def get_var(self):
        return self.var

    def get_std(self):
        return self.std

    def normalize(self, x=None):
        if x is not None:
            self.push(x)
            if self.count <= 1:
                return self.x
            else:
                output= (self.x - self.mean) / self.std
        else:
            output = (self.x - self.mean) / self.std
        return np.clip(output, -self.clip, self.clip)