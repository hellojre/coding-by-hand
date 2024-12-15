from ctypes import util
from turtle import forward
import numpy as np
from initialization import TruncatedNormal
import utils



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
        
        for example_idx, feat_id, feat_val in X.iterate_non_zeros():
            embedding = self._W[feat_id, :]
            output[example_idx,:] += feat_val*embedding #目前比较好奇，为什么是+=,理论上不应是特征*权重 最后求和么？
        return output
    
    def backward(self,prev_grads):
        """
        :param prev_grads: [batch_size, embed_size]
        :return: dw
        """
        dW = {}

        for example_idx, feat_id, feat_val in self._last_input.iterate_non_zeros(): #其实就是forward中输入的X矩阵也就是稀疏特征矩阵
            grad_from_one_example = prev_grads[example_idx, :] * feat_val  #目前写到这里比较好奇，这两个矩阵的维度一样么？一个是EMbedding需要的维度，一个是定义的稀疏特征矩阵维度
        """
        对于每个example，其特征是从整个特征空间中进行标注id
        所以feat_id是可以被重复的
        feat_val理论上应该只有0/1，不确定在这里实现是否是这样
        """
        if  feat_id in dW :  
            dW[feat_id] += grad_from_one_example
        else:
                dW[feat_id] = grad_from_one_example
        return dW
    
class EmbeddingCombineLayer:
        
    def __init__(self,vocab_infos):
        """
        :param vocab_infos: a list of tuple, each tuple is (vocab_name, vocab_size, embed_size)
        """
        self._weights = {}
        #一下初始化是作者参考TensorFlow的初始化 我并没有搞懂
        for vocab_name, vocab_size, embed_size in vocab_infos:
            stddev = 1 / np.sqrt(embed_size) #好像是根号下dK/1
            initialization = TruncatedNormal(
                mean=0,
                stddev=stddev,
                lower=-2,
                upper=2
            )
            self._weights[vocab_name] = initialization(shape=(vocab_size, embed_size))
        # 注意，由于embedding input的稀疏性，一次回代时，不太可能对所有embedding weight有梯度
        # 而是只针对某个field的embedding weight中某feature id对应的行有梯度
        # _grads_to_embed是一个dict,
        # key是"vocab_name@feature_id"的形式，value是一个[embed_size]的ndarray。
        # 因为vocab的weight是多个field所共享的，所以value是每个field对vocab_name@feature_id的梯度的叠加
        self._grads_to_embed = {}
        self._embed_layers = [] #整个的EMbedding layer层     
    
    def add_embedding(self, vocab_name, field_name):  #这一部分的功能就是构建对象
        weight = self._weights[vocab_name]
        layer = EmbeddingLayer(W=weight, vocab_name=vocab_name, field_name=field_name)
        self._embed_layers.append(layer)
        
    @property
    def output_dim(self):
        return sum(layer.output_dim for layer in self._embed_layers)

    def forward(self, sparse_inputs):
        """
        :param sparse_inputs: dict {field_name: SparseInput}
        :return:    每个SparseInput贡献一个embedding vector，返回结果是这些embedding vector的拼接
                    拼接顺序由add_embedding的调用顺序决定
        """
        embedded_outputs = []
        for embed_layer in self._embed_layers:
            sp_input = sparse_inputs[embed_layer.field_name]
            embedded_outputs.append(embed_layer.forward(sp_input))
        
        return np.hstack(embedded_outputs)
    
    def backward(self,prev_grads):
        """
        :param prev_grads:  [batch_size, sum of all embed-layer's embed_size]
                            上一层传入的, Loss对本层输出的梯度
        """
        assert prev_grads.shape[1] == self.output_dim
        col_sizes = [layer.output_dim for layer in self._embed_layers]
        prev_grads_splits = utils.split_column(prev_grads, col_sizes) #按照batch分离 梯度  按层分的，【batch.layer_1,*]
        self._grads_to_embed.clear()

        for layer , layer_prev_grads in zip(self._embed_layers,prev_grads_splits):
            """
            所以在这里，是一个batch的layer_i中的梯度一起更新计算
            """
            layer_grads_to_embed = layer.backward(layer_prev_grads)
            """
            在embedding中，我们定义了backward来计算梯度，其中用的dW字典{feature_id,grads}来保存每个层
            """
            for feature_id , grads in layer_grads_to_embed.items():
                key = "{}@{}".format(layer.vocab_name, feature_id)

                if key in self._grads_to_embed:
                    self._grads_to_embed[key]+=grads
                else:
                    self._grads_to_embed[key] = grads
            """
            ？？？12.7
            在这里看是和我前面的判断有冲突，key是layer.vocab_name.feature_id，那就是说每一个层是分开计算的，并不是统一计算
            难道一个batch内同一层同一个feature的grads并不一起计算么
            """

        
    @property
    def variables(self):
        """ 优化变量
        :return: dict from vocab_name to weight matrix
        """
        return self._weights

    @property
    def grads2var(self):
        """ Loss对优化变量的梯度
        :return: dict, key是"vocab_name@feature_id"的形式，value是一个[embed_size]的ndarray
        """
        return self._grads_to_embed

    @property
    def l2reg_loss(self):
        return 0  # 出于保持稀疏的考虑，在embedding层暂不支持正则








         
