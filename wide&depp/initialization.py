import numpy as np
import scipy.stats

class TruncatedNormal: #z-score
    def __init__(self,mean,stddev,lower, upper):
        """
        参数 (lower - mean) / stddev 和 (upper - mean) / stddev 是截断点的标准化值（即转换为标准正态分布的 Z 分数）。
        loc=mean 和 scale=stddev 分别设置分布的均值和标准差。
        """
        self._rand = scipy.stats.truncnorm(
            (lower - mean) / stddev,
            (upper - mean) / stddev,
            loc=mean,
            scale=stddev)
    
    def __call__(self, shape):
        return self._rand.rvs(size=shape)