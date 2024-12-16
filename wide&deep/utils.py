import logging

def split_column(m,col_sizes): #col_sizes[dense_layer,embedding_layer] m=prev_grads 按照梯度维度将EMbedding分割
    split = []
    start = 0
    for col in col_sizes:
        split.append(m[:,start:(start+col)])
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

def config_logging(fname):
    logging.basicConfig(level=logging.INFO, format='%(message)s')  # re-format to remove prefix 'INFO:root'

    fh = logging.FileHandler(fname)
    fh.setLevel(logging.INFO)
    logging.getLogger("").addHandler(fh)