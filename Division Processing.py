import pandas as pd
import numpy as np
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.multioutput import MultiOutputRegressor
import optuna

from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict
from sklearn.metrics import mean_absolute_error
import shap

from lightgbm import LGBMRegressor, LGBMClassifier
import lightgbm as lgb

from tqdm import tqdm

from sklearn.ensemble import RandomForestRegressor
import warnings ; warnings.filterwarnings('ignore')
import time
from sklearn.metrics import f1_score, roc_auc_score, classification_report


train = pd.read_csv('train.csv', index_col = 'id')
test = pd.read_csv('test.csv', index_col = 'id')
submission = pd.read_csv('sample_submission.csv', index_col='id')

print(train.shape, test.shape, submission.shape)
# (10000, 75) (10000, 71) (10000, 4)

submission.head
train.head


feature_names = list(test)
target_names=list(submission)

x_train = train[feature_names]
x_test = test[feature_names]

y_train = train[target_names]
y_train_hhb = y_train['hhb'] 
y_train_hbo2 = y_train['hbo2']
y_train_ca = y_train['ca']
y_train_na = y_train['na']

parameters
d_train = lgb.Dataset(x_train, label=y_train)
params={'max_depth':10, 'learning_rate':0.07, 'num_iterations':200,
        'boosting':'gbdt','sub_feature':0.5, 'feature_fraction':0.7, 'objective':11,
        'num_leaves':90}

multi_model = MultiOutputRegressor(params)

## - ing
# k-fold cv로 모델 평가해보기, 

