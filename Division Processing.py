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

train = pd.read_csv('train.csv', index_col='id')
test = pd.read_csv('test.csv', index_col='id')
submission = pd.read_csv('sample_submission.csv', index_col='id')


print(train.shape, test.shape, submission.shape)
# (10000, 76) (10000, 72) (10000, 6)

submission.head
train.head

# 결측치 처리 
train.isna().sum().plot
test.isna().sum().plot
plt.show()


test = test.fillna(train.mean())
train = train.fillna(train.mean())

train.head

# train test 분리 

feature_names = list(test)
target_names=list(submission)


x_train = train[feature_names]
x_test = test[feature_names]

y_train = train[target_names]

# t_test 대신 매개변수를 분리하여 개별측정 
y_train_hhb = y_train['hhb'] 
y_train_hbo2 = y_train['hbo2']
y_train_ca = y_train['ca']
y_train_na = y_train['na']



#parameters
d_train = lgb.Dataset(x_train, label=y_train)
params={'max_depth':10, 'learning_rate':0.07, 'num_iterations':200,
        'boosting':'gbdt','sub_feature':0.5, 'feature_fraction':0.7, 'objective':11,
        'num_leaves':90}


base_params={'n_estimators': 900, 'num_leaves': 90, 'learning_rate': 0.1, 
             'colsample_bytree': 0.8, 'subsample': 0.9, 
             'reg_alpha': 5, 'reg_lambda': 7}

#[Ref. http://machinelearningkorea.com/2019/09/29/lightgbm-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0/] 

base_model=LGBMRegressor(objective='l1', subsample_freq=1, silent=False, random_state=18, 
                         importance_type='gain', **base_params)

model=MultiOutputRegressor(base_model)

model.fit(x_train, y_train)

preds = model.predict(x_test)
submission.columns
md = DataFrame(preds, columns = ['hhb','hbo2','ca','na'])
dataframe = pd.DataFrame(md)

dataframe = dataframe.fillna(0)

dataframe.to_csv('predstest.csv')

          
          y_train.values.i


## - ing
# k-fold cv로 모델 평가해보기, 

# y를 분할했으니까 neural network 에 각각 형태에 따라 서로다른 형태로 가공하기위해 더미변수가 필요하다
# 설명변수는 더미데이터로 들어갈 필요가없
# 1. 결측치 주의
# 2. 인과관계 역으로 흘러가지않게 파악(x가 원인, y가 결과, y의 결과가 x에 미치지않게)
# 3. 수치(비율조절 조심)
# xgboost
# xgb = XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=4)
# xgb.fit(x_train, y_train)
