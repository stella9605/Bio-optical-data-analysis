import pandas as pd
import numpy as np
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
run profile1

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

from sklearn.metrics import mean_absolute_error

train = pd.read_csv('train.csv', index_col = 'id')
test = pd.read_csv('test.csv', index_col = 'id')
submission = pd.read_csv('sample_submission.csv', index_col='id')

mean(is.na(train))
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

#parameters
d_train = lgb.Dataset(x_train, label=y_train)
params={'max_depth':10, 'learning_rate':0.07, 'num_iterations':200,
        'boosting':'gbdt','sub_feature':0.5, 'feature_fraction':0.7, 'objective':11,
        'num_leaves':90}
#[Ref. http://machinelearningkorea.com/2019/09/29/lightgbm-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0/] 
multi_model = MultiOutputRegressor(params)

y_train.values.i


## - ing
# k-fold cv로 모델 평가해보기, 

# y를 분할했으니까 neural network 에 각각 형태에 따라 서로다른 형태로 가공하기위해 더미변수가 필요하다
# 설명변수는 더미데이터로 들어갈 필요가없
# 1. 결측치 주의
# 2. 인과관계 역으로 흘러가지않게 파악(x가 원인, y가 결과, y의 결과가 x에 미치지않게)
# 3. 수치(비율조절 조심)
