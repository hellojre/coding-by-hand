from turtle import forward
import numpy as np

class DenseInputCoDenseInputCombineLayer:
    def __init__(self,field_sizes):
        # field_sizes: a list of tuple
        # tuple[0]: field name
        # tuple[1]: input dim for this field
        self._field_sizes = field_sizes
    
    @property
    def output_dim(self):
        return sum(outdim for _,outdim in self._field_sizes)
    
    def forward(self,inputs):
        """
        按field_sizes的顺序从inputs提取ndarray，并拼接起来
        :param inputs: dict of {field_name: ndarray}
        """
        outputs = []
        for field_name,in_dim in self._field_sizes:
            a_input = np.asarray(input[field_name])#转为np对象
            assert in_dim == a_input[1]
            outputs.append(a_input)
        return np.hstack(outputs)#返回一个一维数组，按列
    

class SparseInput:
    """
    目前决定采用3个list的方式来表示一个理论、稠密形状为[batch_size, max_bag_size]的稀疏输入
    所谓max_bag_size，是一个理论概念，可以认为infinity，在代码中并不出现，也不会对代码造成限制
    比如表示用户行为历史，max_bag_size可以是用户一段历史内阅读的文章数、购买的商品数
    比如表示用户的手机使用习惯，max_bag_size可以是所有app的数目
    这里，我们将这些信息表示成一个bag，而不是sequence，忽略其中的时序关系

    第一个list example_indices: 是[n_non_zeros]的整数数组，表示在[batch_size, max_bag_size]中的行号（样本序号），>=0 and < batch_size
                               而且要求其中的数值是从小到大，排好序的
    第二个list feature_ids:     是[n_non_zeros]的整数数组，表示非零元对应特征的序号，可以重复
    第三个list feature_values:  是[n_non_zeros]的浮点数组，表示非零元对应特征的数值
    举例来说，第i个非零元(0<=i<n_non_zeros)
    它对应哪个样本？example_indices[i]
    它对应哪个特征？feature_ids[i]
    它的数值是多少？values[i]
    """

    def __init__(self, n_total_examples, example_indices, feature_ids, feature_values):
        assert len(example_indices) == len(feature_ids) == len(feature_values) #目前写到这里我是比较好奇的 为什么需要三者长度一样？
        self._example_indices = example_indices
        self._feature_ids = feature_ids
        self._feature_values = feature_values

        self._n_total_examples = n_total_examples # 理论上这个batch包含的样本的个数，相当于SparseTensor中的dense_shape[0]
        self.__nnz_idx = 0 #未知作用

    def add(self, example_idx, feat_id, feat_val):
        self._example_indices.append(example_idx)
        self._feature_ids.append(feat_id)
        self._feature_values.append(feat_val)
    
    def iterate_non_zeros(self):
        return zip(self._example_indices,self._feature_ids,self._feature_values) #返回稀疏特征对象，一个三元组
    
    def __move_to_next_example(self, nnz_idx):
        """
        返回当前样本的所有feature id和feature value
        并把nnz_index移动到下一个样本的起始位置
        """
        if nnz_idx > len(self._example_indices):
            return None

        end = nnz_idx+1 #nnz_idx是某个item稀疏特征在稀疏矩阵的起始地址
        while end < len(self._example_indices) and self._example_indices[end] == self._example_indices[nnz_idx]:
            current_feat_ids = self._feature_ids[nnz_idx:end]
            current_feat_vals = self._feature_values[nnz_idx:end]

        return end,current_feat_ids,current_feat_vals
    
    def get_example_in_order(self, example_idx):
        """
        :param example_idx: 有一个前提，example_idx必须从0到batch_size顺序输入
        :return: 与example_idx对应的feat_ids和feat_vals
        """ 

        if self.__nnz_idx >= len(self._example_indices):
            return [],[]
        elif self._example_indices[self.__nnz_idx] == example_idx:
            self.__nnz_idx,current_feat_ids,current_feat_vals = self.__move_to_next_example(self.__nnz_idx)
            return current_feat_ids,current_feat_vals
        elif self._example_indices[self.__nnz_idx] > example_idx:
            # 等待调用者下次传入更大的example_idx
            return [], []
        else:
            # 如果当前example_index并不是调用者需要的example_idx
            # 则一定是比外界需要用example_idx大，等待调用者传入更大的example_idx
            # 如果比外界需要用example_idx小，说明调用方式不对
            raise ValueError("incorrect invocation")

        """
        有点好奇get_example_in_order是干嘛用的，相当于只要部分特征，不是很懂
        """



