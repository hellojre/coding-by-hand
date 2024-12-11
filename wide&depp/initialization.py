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


class Zero:   
    def __call__(self,shape):
        return np.zeros(shape)



class GlorotUniform:
    def __call__(self,shape):
        fan_in,fan_out = shape
        scale = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-scale,scale,shape)
    

class GlorotNormal:
    def __call__(self, shape):
        fan_in, fan_out = shape
        stdev = np.sqrt(2 / (fan_out + fan_in))
        return np.random.normal(loc=0, scale=stdev, size=shape)


_Global_Initializers = {} # initializers which can be shared

    
def get_global_init(name):
    if name in _Global_Initializers:
        return _Global_Initializers[name]

    if name == "zero":
        initializer = Zero()
    elif name == "glorot_uniform":
        initializer = GlorotUniform()
    elif name == "glorot_normal":
        initializer = GlorotNormal()
    else:
        raise ValueError('unknown initializer={}'.format(name))

    _Global_Initializers[name] = initializer
    return initializer
    

