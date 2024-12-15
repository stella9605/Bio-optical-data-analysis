## ca

n_estimators = [100,200,400,800,1600,2000]
max_features = ['auto', 'sqrt', 'log2']
max_depth = [10,20,30,40,50]
max_depth.append(None)
min_samples_split  = [2,5,10,15,20]
min_samples_leaf = [1,2,5,10,15]

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


grid_search = GridSearchCV(m_rf_ca, param_grid, cv =5,
                            scoring='neg_mean_absolute_error',
                            return_train_score=True)

grid_search.fit(train_x, list(train_y['ca']))

# n_estimators 의 값이 작을수록 세밀한 탐색이 가능하다

grid_search.best_params_
# {'max_features': 8, 'n_estimators': 30}

grid_search.best_estimator_
#RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                      # max_depth=None, max_features=8, max_leaf_nodes=None,
                      # max_samples=None, min_impurity_decrease=0.0,
                      # min_impurity_split=None, min_samples_leaf=1,
                      # min_samples_split=2, min_weight_fraction_leaf=0.0,
                      # n_estimators=30, n_jobs=None, oob_score=False,
                      # random_state=None, verbose=0, warm_start=False)


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-mean_score), params)
    



m_rf_ca = rf_r(n_estimators = 2000,
               max_features = 30,
               max_depth = 30,
               min_samples_split = 5,  # 5
               min_samples_leaf = 4,   # 1
               n_jobs = -1,
               random_state = 99)
m_rf_ca.fit(train_x, list(train_y['ca']))

%timeit

test_y_ca_predict = m_rf_ca.predict(test_x)
train_y_ca_predict = m_rf_ca.predict(train_x)

param_grid =[
    {'n_estimators':n_estimators, 'max_features':max_features,
     'max_depth':max_depth, 'min_samples_split':min_samples_split,'min_samples_leaf':min_samples_leaf }]
rf_r_random = RandomizedSearchCV(estimator = m_rf_ca, 
                                 param_distributions = param_grid,
                                 n_iter=500,
                                 cv = 5,
                                 verbose = 2,
                                 random_state = 42,
                                 n_jobs = -1)
rf_r_random.fit(train_x, train_y['ca'])

# RandomizedSearchCV(cv=5, error_score=nan,
#                    estimator=RandomForestRegressor(bootstrap=True,
#                                                    ccp_alpha=0.0,
#                                                    criterion='mse',
#                                                    max_depth=30,
#                                                    max_features=30,
#                                                    max_leaf_nodes=None,
#                                                    max_samples=None,
#                                                    min_impurity_decrease=0.0,
#                                                    min_impurity_split=None,
#                                                    min_samples_leaf=4,
#                                                    min_samples_split=5,
#                                                    min_weight_fraction_leaf=0.0,
#                                                    n_estimators=2000, n_jobs=-1,
#                                                    oob_score=False,
#                                                    rando...
#                                                    warm_start=False),
#                    iid='deprecated', n_iter=500, n_jobs=-1,
#                    param_distributions=[{'max_depth': [10, 20, 30, 40, 50,
#                                                        None],
#                                          'max_features': ['auto', 'sqrt',
#                                                           'log2'],
#                                          'min_samples_leaf': [1, 2, 5, 10, 15],
#                                          'min_samples_split': [2, 5, 10, 15,
#                                                                20],
#                                          'n_estimators': [100, 200, 400, 800,
#                                                           1600, 2000]}],
#                    pre_dispatch='2*n_jobs', random_state=42, refit=True,
#                    return_train_score=False, scoring=None, verbose=2)

# train_y_ca_rdpredict = rf_r_random.fit(train_x)
# Fitting 5 folds for each of 500 candidates, totalling 2500 fits
# [Parallel(n_jobs=-1)]: Using backend LokyBackend with 12 concurrent workers.
# [Parallel(n_jobs=-1)]: Done  17 tasks      | elapsed:    1.9s
# [Parallel(n_jobs=-1)]: Done 492 tasks      | elapsed:    2.6s
# [Parallel(n_jobs=-1)]: Done 2453 tasks      | elapsed:    3.9s
# [Parallel(n_jobs=-1)]: Done 2500 out of 2500 | elapsed:    4.0s finished



train_y_ca_rdpredict = rf_r_random.fit(train_x, train_y['ca'])
test_y_ca_rdpredict = rf_r_random.fit(test_x, test_y['ca'])


##MAE 
mean_absolute_error(train_y['ca'], train_y_ca_rdpredict)   
mean_absolute_error(test_y['ca'], test_y_ca_rdpredict) 
