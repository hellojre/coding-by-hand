
from turtle import forward
import numpy as np

class Sigmod:
    def __init__(self):
        self._last_forward_result = None

    def forward(self,X):
        self._last_forward_result = 1.0/(1+np.exp(-X))
        return self._last_forward_result
    
    def backward(self,prev_grads):
        assert self._last_forward_result.shape == prev_grads.shape

        return prev_grads*self._last_forward_result*(1-self._last_forward_result) #后面的是sigmod的导数
    

class ReLU:
    def __init__(self):
        self._last_input = None

    def forward(self,X):
        self._last_input = X
        return np.maximum(0, X)
    
    def backward(self,prev_grads):
        assert prev_grads.shape == self._last_input.shape
        local_grads = np.zeros_like(self._last_input)
        local_grads[self._last_input > 0] = 1.0

        return prev_grads * local_grads