from turtle import forward
import numpy as np


class EmbeddingLayer:

    """
    __init 函数会自动接收EMbeddingLayer传来的参数
    如果其他函数不使用接受来的参数，可以不用加self，否则则需要加self
    """
    def __init__(self,W,vocab_name, field_name):
        """
        :param W: dense weight matrix, [vocab_size,embed_size]
        :param b: bias, [embed_size]
        """
        self.vocab_name = vocab_name
        self.field_name = field_name
        self._W = W
        self._last_input = None

    @property #未加参数需要添加该标注
    def output_dim(self):
        return self._W.shape[1]
    
    def forward(self,X):
        """
        :param X: SparseInput
        :return: [batch_size, embed_size]
        """
        self._last_input = X
        output = np.zeros((X.n_total_examples, self._W.shape[1]))
        
