import pandas as pd
import numpy as np
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.model_selection import KFold
from xgboost import plot_importance
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings(action='ignore')

train = pd.read_csv('train.csv', index_col ='id')
test = pd.read_csv('test.csv', index_col = 'id')
submission=pd.read_csv('sample_submission.csv', index_col = 'id')

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
sns.heatmap(train.corr().loc['rho':'990_src','hhb':].abs())
sns.heatmap(train.corr().loc['650_dst':'990_dst','hhb':].abs())
sns.heatmap(train.corr().loc['rho':'990_dst','hhb':].abs())



X = train.loc[:,'rho':'990_dst']
Y = train.loc[:,'hhb':'na']

x_train, y_train, x_test, y_test = train_test_split(X,
                                                    Y,
                                                    random_state=0)
print(x_train.shape, x_test.shape)
# (7500, 71) (7500, 4)

xgb = XGBClassifier(n_estimators=500,learning_rate = 0.1, max_depth=4 )
xgb.fit(x_train,y_train)
xgb.predict(x_test)



# xgboost 로 kfold, 
# train set 분리하고 params로 lightgbm 사용할것 


# 흡광도 
# 계산식 (흡광도 계산식)
# A(흡광도) = -log10(I(투과방사선)/I0(입사방사선))  
#           = ε(흡광계수) ⋅ b(투과 경로 길이(cm)) ⋅ c(농도)
train.head

# 튜닝 함수
def tuning_var(s):
    s_rho = s[0]          # _rho
    s_src = s[1:36]       # _src
    s_dst = s[36:]        # _dst    

    # index 표준화
    set_index = s_src.index.str.split('_').str[0]
    s_src.index = set_index
    s_dst.index = set_index

    # 계산식 (흡광도 계산식)
    # A(흡광도) = -log10(I(투과방사선)/I0(입사방사선))  
    #           = ε(흡광계수) ⋅ b(투과 경로 길이(cm)) ⋅ c(농도)
    s_ds_st = ((s_dst / s_src) / (s_rho/10))
    
    # 계산 완료후 inf,nan 0으로 치환
    s_ds_st = [i if i != np.inf else 0.0 for i in s_ds_st ]
    s_ds_st = Series(s_ds_st).fillna(value = 0)
    
    # math.log 계산을 위해 0을 1로 치환후 계산(흡광계수는 1로 가정한다.)
    s_ds_st = [1 if i == 0 else i for i in s_ds_st ]
    
    # 변수 튜닝 반환
    out_s = Series(map(lambda x : -math.log(x,10), s_ds_st))
    out_s.index= set_index
    return(out_s)

tunning_train_x = train_x.apply(tuning_var, axis = 1)


# xgboost

