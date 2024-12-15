
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
