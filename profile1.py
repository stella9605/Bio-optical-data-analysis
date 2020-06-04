import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series
from math import trunc
from numpy import nan as NA

import re
import matplotlib.pyplot as plt

# 데이터 셋
from sklearn.datasets import load_iris as iris
from sklearn.datasets import load_breast_cancer as cancer



from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.neighbors import KNeighborsRegressor as knn_r

from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.tree import DecisionTreeRegressor as dt_r

from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import RandomForestRegressor as rf_r

from sklearn.ensemble import GradientBoostingClassifier as gb
from sklearn.ensemble import GradientBoostingRegressor as gb_r

# scale

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

# CV

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
