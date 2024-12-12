import numpy as np
from input_layer import DenseInputCombineLayer
from embedding import EmbeddingCombineLayer
from dense_layer import DenseLayer
from activation import ReLU
import utils
import logging
from collections import namedtuple
from base_estimator import BaseEstimator
