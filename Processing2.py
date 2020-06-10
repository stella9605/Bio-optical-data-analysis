import pandas as pd
import numpy as np
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings(action='ignore')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission=pd.read_csv('sample_submission.csv')

train.head()  # (5 x 76)
test.head()   # (5 x 72)

# 결측치 처리 
train.isna().sum().plot
test.isna().sum().plot
plt.show()


test = test.fillna(train.mean())
train = train.fillna(train.mean())


# 데이터와 hho, hbo2, ca, na 상관관계 분석
plt.figure(figsize=(4,12))
sns.heatmap(train.corr().loc['rho':'990_dst','hhb':].abs())

x_train = train.loc[:,'650_dst':'990_dst']
y_train = train.loc[:,'hhb':'na']
x_train.shape # (10000,35)
y_train.shape # (10000,4)

# xgboost 로 kfold, 
# train set 분리하고 params로 lightgbm 사용할것 