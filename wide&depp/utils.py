import logging

def split_column(m,col_sizes):
    split = []
    start = 0
    for col in col_sizes:
        split.append[m[:,start:(start+col)]]
        start+=col
    
    assert start == m.shape[1]
    return split

def chunk(stream,chunk_size):
    buf = []
    
    for item in stream:
        buf.append(item)
        if len(buf) >= chunk_size:
            yield buf
            del buf[:]
    
    if len(buf) > 0:
        yield buf