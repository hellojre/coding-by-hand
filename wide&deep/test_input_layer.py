import numpy as np
from input_layer import SparseInput

def test_get_example_in_order_from_sparse(example_indices, batch_size):
    sp_input = SparseInput(example_indices=example_indices,
                           feature_ids=example_indices,
                           feature_values=example_indices,
                           n_total_examples=batch_size)

    for example_idx in range(batch_size):
        feat_ids, feat_vals = sp_input.get_example_in_order(example_idx)
        print("\n**************** {}-th example: ".format(example_idx))
        print("feature ids:    {}".format(feat_ids))
        print("feature values: {}".format(feat_vals))





if __name__ == "__main__":
    test_get_example_in_order_from_sparse(example_indices=[1, 1, 1, 3, 4, 6],batch_size=10)


"""
**************** 0-th example: 
feature ids:    []
feature values: []

**************** 1-th example: 
feature ids:    [1, 1, 1]
feature values: [1, 1, 1]

**************** 2-th example: 
feature ids:    []
feature values: []

**************** 3-th example: 
feature ids:    [3]
feature values: [3]

**************** 4-th example: 
feature ids:    [4]
feature values: [4]

**************** 5-th example: 
feature ids:    []
feature values: []

**************** 6-th example: 
feature ids:    [6]
feature values: [6]

**************** 7-th example: 
feature ids:    []
feature values: []

**************** 8-th example: 
feature ids:    []
feature values: []

**************** 9-th example: 
feature ids:    []
feature values: []
"""