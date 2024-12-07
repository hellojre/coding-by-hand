import pandas as pd
import sys
import numpy as np
import random
import utils
from input_layer import SparseInput
import bisect
import argparse
from tqdm import tqdm

"""
根据分析raw_test.txt  具有一下类别 其实数值型如：age不做稀疏矩阵处理，但比如education这种类别（稀疏特征）需要构建EMbedding 所以要建立词表
age,workclass,fnlwgt,education,education_num,
marital_status,occupation,relationship,race,gender,
capital_gain,capital_loss,hours_per_week,native_country,income_bracket
"""

VOCAB_LISTS = {
    'education': ['Bachelors',
                  'HS-grad',
                  '11th',
                  'Masters',
                  '9th',
                  'Some-college',
                  'Assoc-acdm',
                  'Assoc-voc',
                  '7th-8th',
                  'Doctorate',
                  'Prof-school',
                  '5th-6th',
                  '10th',
                  '1st-4th',
                  'Preschool',
                  '12th'],

    'marital_status': ['Married-civ-spouse',
                       'Divorced',
                       'Married-spouse-absent',
                       'Never-married',
                       'Separated',
                       'Married-AF-spouse',
                       'Widowed'],

    'relationship': ['Husband',
                     'Not-in-family',
                     'Wife',
                     'Own-child',
                     'Unmarried',
                     'Other-relative'],

    'workclass': ['Self-emp-not-inc',
                  'Private',
                  'State-gov',
                  'Federal-gov',
                  'Local-gov',
                  'Self-emp-inc',
                  'Without-pay',
                  'Never-worked'],

    'occupation': ['Tech-support',
                   'Craft-repair',
                   'Other-service',
                   'Sales',
                   'Exec-managerial',
                   'Prof-specialty',
                   'Handlers-cleaners',
                   'Machine-op-inspct',
                   'Adm-clerical',
                   'Farming-fishing',
                   'Transport-moving',
                   'Priv-house-serv',
                   'Protective-serv',
                   'Armed-Forces']
}

VOCAB_MAPPINGS = {field:{featname: idx for featname,idx in enumerate(featnames)} 
                  for field,featnames in VOCAB_LISTS.items()}

AGE_BOUNDARIES = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65]

DENSE_FIELDS = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']

#均值和标准差
DENSE_LOG_MEAN_STD = {'age': (3.6183599219864133, 0.35003117354646957),
                      'education_num': (2.372506496597371, 0.27381608590073075),
                      'capital_gain': (0.7346209104536965, 2.4547377400238553),
                      'capital_loss': (0.35030508122367104, 1.5845809727578963),
                      'hours_per_week': (3.665366478972777, 0.38701441353280025)}

CATEGORY_FIELDS = ['education', 'marital_status', 'relationship', 'workclass', 'occupation', 'age_buckets']