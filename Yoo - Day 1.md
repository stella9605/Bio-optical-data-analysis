#### *데이터 전처리
```python
run profile1

train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)

train.head()

train.isnull().sum()[train.isnull().sum().values > 0]

# 결측치 보완
train_dst = train.filter(regex='_dst$', axis=1).replace(0, np.NaN) # dst 데이터만 따로 뺀다.
test_dst = test.filter(regex='_dst$', axis=1).replace(0, np.NaN) # 보간을 하기위해 결측값을 삭제한다.
test_dst.head(1)

train_dst = train_dst.interpolate(methods='linear', axis=1)
test_dst = test_dst.interpolate(methods='linear', axis=1)
# 스팩트럼 데이터에서 보간이 되지 않은 값은 0으로 일괄 처리한다.
train_dst.fillna(0, inplace=True) 
test_dst.fillna(0, inplace=True)
test_dst.head(1)

train.update(train_dst) # 보간한 데이터를 기존 데이터프레임에 업데이트 한다.
test.update(test_dst)
```

- 위코드까지는 사이트 코드공유에서 퍼와서 실행함 오늘 수업필기자료보면 밑에 코딩내용 그대로나와있어요 보시면되겠고 저는 거기에서 train,test 분류하고 변수 재설정하고 모델만 조금 수정했습니다 

#### *ANN 모델 학습을 통한 예측모델 생성 및 평가
```
# 데이터 분류
dataset = train.values 

X = dataset[:,0:-4].astype(float) 
Y_obj = dataset[:,-4:]

train_x,test_x,train_y,test_y=train_test_split(X,
                                               Y_obj,
                                               random_state=0)

# 모델의 설정
model = Sequential() 
model.add(Dense(71, input_dim=71, activation='relu')) 
model.add(Dense(35, activation='relu')) 
model.add(Dense(17, activation='relu')) 
model.add(Dense(4, activation='softmax'))
 
# 모델 컴파일  
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy']) 
 
# 모델 실행 
model.fit(train_x, train_y, epochs=100, batch_size=10) 
=> 결과
# Epoch 100/100
# 7500/7500 [==============================] - 1s 87us/step - 
# loss: 44.0200 - accuracy: 0.5888

 
# 결과 출력  
print("\n Accuracy: %.4f" % (model.evaluate(test_x,test_y)[1]))
=> 결과
#  Accuracy: 0.6084
```
-  결론 :  ANN 모델링을 사용한 이유는 4개의 종속변수인 각 물질의 농도가 모든  
       설명변수인 각 광원 및 흡광스펙트럼 값에 영향을 주는데 이 독립변수들의 
       결합을 통해 종속변수를 설명하기 위해서입니다.
       선생님께서 말씀하셨듯이 모델을 튜닝하고 변경하면서 높은 예측력을 기대할
       것이 아니라 의미있는 설명변수를 찾고 데이터자체를 가공하는 것에 집중해야
       하지 않을까 생각되네요~~~ 
  
