import logging

def split_column(m,col_sizes):
    split = []
    start = 0
    for col in col_sizes:
        split.append[m[:,start:(start+col)]]
        start+=col
    
    assert start == m.shape[1]
    return split