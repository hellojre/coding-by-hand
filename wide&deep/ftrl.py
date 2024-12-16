import numpy as np
from collections import defaultdict

class FtrlEstimator:
    """
    每个field对应一个Ftrlsimator,类比在于TensorFlow WDL中 一个feature column对应一个FtrlEstimator
    """


    def __init__(self,alpha,beta,L1,L2):
        self._alpha = alpha
        self._beta = beta
        self._L1 = L1
        self._L2 = L2
        #设置一个默认字典，空时以及不存在时为0.0
        self._n = defaultdict(float) #每个特征出现的次数
        self._z = defaultdict(float) #每个特征的累计梯度

        # lazy weights, 实际上是一个临时变量，只在：
        # 1. 对应的feature value != 0, 并且
        # 2. 之前累积的abs(z) > L1
        # 两种情况都满足时，w才在feature id对应的位置上存储一个值
        # 而且w中数据的存储周期，只在一次前代、回代之间，在新的前代开始之前，就清空上次的w
        self._w = {}

        self._current_feat_ids = None
        self._current_feat_vals = None

    

    def predict_logit(self,feature_ids,feature_values):
        """
        :param feature_ids: non-zero feature ids for one example
        :param feature_values: non-zero feature values for one example
        :return: logit for this example
        """
        self._current_feat_ids = feature_ids
        self._current_feat_vals = feature_values

        logit = 0
        self._w.clear()# lazy weights，所以没有必要保留之前的weights


        # 如果当前样本在这个field下所有feature都为0，则feature_ids==feature_values==[]
        # 则没有以下循环，logit=0

        for feat_id,feat_val in zip(feature_ids,feature_values):
            z = self._z[feat_id]
            sign_z = -1. if z<0 else 1.

            # build w on the fly using z and n, hence the name - lazy weights
            # this allows us for not storing the complete w
            # if abs(z) <= self._L1: self._w[feat_id] = 0.  # w[i] vanishes due to L1 regularization

            if abs(z) > self._L1: #当z的绝对值大于l1正则阈值时 就会执行以下代码
                #apply prediction time L1, L2 regularization to z and get w
                w = (sign_z * self._L1 - z)/((self._beta + np.sqrt(self._n[feat_id]))/self._alpha + self._L2)
                """
                self._n[feat_id]) 是特征 feat_id 的出现次数的平方根 self._beta 是一个常数防止分母为0
                
                """
                self._w[feat_id] = w
                logit += w*feat_val
            
        return logit 
    

    def update(self,pred_proba,label):
        """
        :param pred_proba:  与last_feat_ids/last_feat_vals对应的预测CTR
                            注意pred_proba并不一定等于sigmoid(predict_logit(...))，因为要还要考虑deep侧贡献的logit
        :param label:       与last_feat_ids/last_feat_vals对应的true label
        """
        grad2logit = pred_proba - label
        # 如果当前样本在这个field下所有feature都为0，则没有以下循环，没有更新
        for feat_id,feat_val in zip(self._current_feat_ids,self._current_feat_vals):
            g = grad2logit * feat_val #将其差值视为梯度
            g2 = g * g
            n = self._n[feat_id]

            self._z[feat_id] += g #更新累积梯度

            if feat_id in self._w: # if self._w[feat_id] != 0
                sigma = (np.sqrt(n+g2) - np.sqrt(n)) / self._alpha
                self._z[feat_id] -= sigma * self._w[feat_id]


            self._n[feat_id] = n + g2