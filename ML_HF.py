import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn import linear_model
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix


'''
1. 匯入資料
2. 資料預處理
'''
hf = pd.read_csv('heart_failure_clinical_records_dataset.csv')
# print(hf)
hf.isnull().any()

# features = [hf['age'],hf['anaemia'],hf['creatinine_phosphokinase'],
#             hf['diabetes'],hf['ejection_fraction'],hf['high_blood_pressure'],
#             hf['platelets'],hf['serum_creatinine'],hf['serum_sodium'],
#             hf['sex'],hf['smoking'],hf['time']]

# for i in features:
#     x = pd.DataFrame([i]).T
#     logistic = linear_model.LogisticRegression()
#     logistic.fit(x,y)
#     print(x.columns)
#     print('截距=' ,logistic.intercept_)
#     print('迴歸係數=',logistic.coef_)
#     print('正確率', logistic.score(x, y))
#     print()

'''
3. 特徵選擇
'''

print('='*50 + '\n特徵選擇')
y = hf['DEATH_EVENT']



  # 決定測試資料量

for i in np.linspace(0.2, 0.5, 7):
    x = hf.iloc[:,:11]
    x_train, x_test, y_train, y_test = tts(x, y, test_size=i, random_state=1)
    logistic = linear_model.LogisticRegression(max_iter=10000)
    logistic.fit(x_train,y_train)
    pre = logistic.predict(x_test)
    print('\n測試資料集佔',i)
    print('score_all =',logistic.score(x, y))
    print('score_split =',logistic.score(x_train, y_train))
    print('accuracy_split =', accuracy_score(y_test, pre))
    
# for column in hf.iloc[:,:11]:
#     x = pd.DataFrame([hf[column]]).T
#     logistic = linear_model.LogisticRegression()
#     logistic.fit(x,y)
#     print(x.columns)
#     print('截距=' ,logistic.intercept_)
#     print('迴歸係數=',logistic.coef_)
#     print('正確率', logistic.score(x, y))
#     if logistic.score(x, y) > 0.68:
#         print('正確率>0.68')
#     print()

 # print score > 0.68 only
for column in hf.iloc[:,:11]:
    x = pd.DataFrame([hf[column]]).T
    logistic.fit(x,y)
    pre = logistic.predict(x)
    if logistic.score(x, y) > 0.68:
        print(x.columns)
        print('Score', logistic.score(x, y))
        print('accuracy =', accuracy_score(y, pre))
    print()
    
 # 將以上述方式找到的五個欄位帶入計算正確率(不分割資料)
x = pd.DataFrame([hf['age'],hf['creatinine_phosphokinase'],
                  hf['ejection_fraction'],hf['serum_creatinine'],
                  hf['serum_sodium']]).T

logistic.fit(x, y)
print('正確率',logistic.score(x, y))

  # 用SelectKBest做資料選擇：chi2
print('-'*40 + '\nchi2')
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
x = pd.DataFrame(hf.iloc[:,:11])
y = hf['DEATH_EVENT']

selector = SelectKBest(chi2, k=5)
selector.fit(x, y)
x_new = selector.transform(x)

x.columns[selector.get_support(indices=True)]
new_col = list(x.columns[selector.get_support(indices=True)])
print(new_col)

# print('-'*40 + '\nchi2')
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# x = pd.DataFrame(hf.iloc[:,:11])
# y = hf['DEATH_EVENT']
# print(x.shape)
# x_new = SelectKBest(chi2, k=5).fit_transform(x, y)
# print(x_new.shape)

    # 查看正確率
x_train, x_test, y_train, y_test = tts(x_new, y, test_size=0.4, random_state=1)
logistic.fit(x_train,y_train)
pre = logistic.predict(x_test)
print('score_all =',logistic.score(x_new, y))
print('score_split =',logistic.score(x_train, y_train))
print('accuracy_split =', accuracy_score(y_test, pre))

  # 用SelectKBest做資料選擇：f_regression
print('-'*40 + '\nf_regression')
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
x = pd.DataFrame(hf.iloc[:,:11])
y = hf['DEATH_EVENT']

selector = SelectKBest(f_regression, k=5)
selector.fit(x, y)
x_new = selector.transform(x)

x.columns[selector.get_support(indices=True)]
new_col = list(x.columns[selector.get_support(indices=True)])
print(new_col)

# print('-'*40 + '\nf_regression')
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import f_regression
# x = pd.DataFrame(hf.iloc[:,:11])
# y = hf['DEATH_EVENT']
# print(x.shape)
# x_new = SelectKBest(f_regression, k=5).fit_transform(x, y)
# print(x_new.shape)
# print(x_new.columns)

    # 查看正確率
x_train, x_test, y_train, y_test = tts(x_new, y, test_size=0.4, random_state=1)
logistic.fit(x_train,y_train)
pre = logistic.predict(x_test)
print('score_all =',logistic.score(x_new, y))
print('score_split =',logistic.score(x_train, y_train))
print('accuracy_split =', accuracy_score(y_test, pre))

 # 挑出預測結果最佳之組合
print('='*50 + '\n挑出預測結果最佳之組合')
 # 正確率 = 85
x = pd.DataFrame([hf['age'],hf['creatinine_phosphokinase'],
                  hf['ejection_fraction'],hf['serum_creatinine'],
                  hf['high_blood_pressure']]).T

x_train, x_test, y_train, y_test = tts(x, y, test_size=0.4, random_state=1)

logistic.fit(x_train, y_train)
pre = logistic.predict(x_test)
print()
print('正確率',logistic.score(x_test, y_test))
print('accuracy =', accuracy_score(y_test, pre))

 # 正確率 = 81
x = pd.DataFrame([hf['age'],hf['serum_sodium'],
                  hf['ejection_fraction'],hf['serum_creatinine'],
                  hf['high_blood_pressure']]).T

x_train, x_test, y_train, y_test = tts(x, y, test_size=0.4, random_state=1)

logistic.fit(x_train, y_train)
pre = logistic.predict(x_test)
print()
print('正確率',logistic.score(x_test, y_test))
print('accuracy =', accuracy_score(y_test, pre))

 # 正確率 = 85
x = pd.DataFrame([hf['age'], hf['ejection_fraction'],
                  hf['serum_creatinine'], hf['high_blood_pressure']]).T

x_train, x_test, y_train, y_test = tts(x, y, test_size=0.4, random_state=1)

logistic.fit(x_train, y_train)
pre = logistic.predict(x_test)
print()
print('正確率',logistic.score(x_test, y_test))
print('accuracy =', accuracy_score(y_test, pre))


'''
4. 建模與預測

(1) 邏輯迴歸
'''

print('='*50 + '\n邏輯迴歸\n')

 # 不分割資料、不更改欄位數

x = pd.DataFrame(hf.iloc[:,:11])
y = hf['DEATH_EVENT']

 # 建立模型
logistic.fit(x,y)
print('不分割資料、使用11個特徵做預測')
print('accuracy =',logistic.score(x, y))

 # 僅用挑出的4個特徵做預測
x = pd.DataFrame([hf['age'], hf['ejection_fraction'],
                  hf['serum_creatinine'], hf['high_blood_pressure']]).T

x_train, x_test, y_train, y_test = tts(x, y, test_size=0.4, random_state=1)

logistic.fit(x_train, y_train)
pre = logistic.predict(x_test)
# print('正確率',logistic.score(x_test, y_test))
print('-'*40 + '\n測試資料佔40%、使用4個特徵做預測')
print('accuracy =', accuracy_score(y_test, pre))

 # 匯出混淆矩陣
import seaborn as sebrn
import matplotlib.pyplot as plt

sebrn.set(font_scale = 1.8)
plt.figure(figsize=[10,6.6666667])
  # 圖片尺寸要在圖片產生之前先設定好
conf_matrix = (pd.crosstab(y_test, pre))
fx = sebrn.heatmap(conf_matrix, annot=True, cmap='Blues')

fx.set_title('\nConfusion Matrix - Logistic Regression\n')
fx.set_xlabel('Predicted Values')
fx.set_ylabel('Actual Values')

fx.xaxis.set_ticklabels(['0','1'])
fx.yaxis.set_ticklabels(['0','1'])

plt.show()


'''
SBS演算法
'''
from sklearn.base import clone
from itertools import combinations

class SBS():
    def __init__(self, estimator, k_features,
                 scoring = accuracy_score,
                 test_size=0.2, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
    
    def fit(self, x, y):
        x_train, x_test, y_train, y_test = \
            tts(x, y, test_size = self.test_size,
                random_state = self.random_state)
        dim = x_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(x_train, y_train, x_test, y_test,self.indices_)
        self.scores_ = [score]
        
        while dim > self.k_features:
            scores = []
            subsets = []
            
            for p in combinations(self.indices_, r=dim-1):
                score = self._calc_score(x_train, y_train, x_test, y_test, p)
                scores.append(score)
                subsets.append(p)
            
            best  = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]
        
        return self
    
    def transform(self, x):
        return x[:, self.indices_]
    
    def _calc_score(self, x_train, y_train, x_test, y_test, indices):
        self.estimator.fit(x_train[:, indices], y_train)
        y_pred = self.estimator.predict(x_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score


'''
(2) KNN
'''

 # 用SBS選擇特徵

import matplotlib.pyplot as plt
from sklearn import neighbors

print('='*50 + '\nKNN')
print('\nKNN + SBS循序向後選擇')

x = pd.DataFrame(hf.iloc[:,:11])
y = hf['DEATH_EVENT']
x = x.to_numpy()
y = y.to_numpy()
# x.shape, y.shape

x_train, x_test, y_train, y_test = tts(x, y, test_size=0.4, random_state=1)

knn = neighbors.KNeighborsClassifier(n_neighbors=7)
sbs = SBS(knn, k_features=1)
sbs.fit(x_train, y_train)

 # 繪圖
k_feat = [len(k) for k in sbs.subsets_]

plt.figure(figsize=(9,6))
plt.plot(k_feat, sbs.scores_, marker='o', markersize=10)
plt.tick_params(labelsize=17)
plt.ylim([0.6, 0.9])
plt.ylabel('Accuracy', size=17)
plt.xlabel('Number of Features', size=17)
plt.grid()
plt.tight_layout()
plt.show()

k3 = list(sbs.subsets_[8])
print(hf.columns[1:][k3])


 # 實際將結果帶入KNN分類器
from sklearn import neighbors

# x = pd.DataFrame([hf['anaemia'],hf['creatinine_phosphokinase'],hf['high_blood_pressure']]).T
# y = hf['DEATH_EVENT']
# x_train, x_test, y_train, y_test = tts(x, y, test_size=0.5, random_state=1)

# k = 7

# knn = neighbors.KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train[:, k3], y_train)

pred = knn.predict(x_test[:, k3])
print('Score =', knn.score(x_test[:, k3], y_test))
print('accuracy =', accuracy_score(y_test, pred))

 # 繪製混淆矩陣
sebrn.set(font_scale = 1.8)
plt.figure(figsize=[10,6.6666667])
conf_matrix = (pd.crosstab(y_test, pred))
fx = sebrn.heatmap(conf_matrix, annot=True, cmap='Blues')

fx.set_title('\nConfusion Matrix - KNN+SBS\n')
fx.set_xlabel('Predicted Values')
fx.set_ylabel('Actual Values')

fx.xaxis.set_ticklabels(['0','1'])
fx.yaxis.set_ticklabels(['0','1'])

plt.show()

# 第一次執行上面的SBS語法出現錯誤
# 上網查到下面的寫法，但仍顯示錯誤
# 查看資料型態
# from sklearn.datasets import load_wine

# ds = load_wine()
# m = ds.data
# q = ds.target
# m.shape, q.shape

# 發現問題，將df轉成array後重新測試

# 空白圖片，尚未解決
# x = pd.DataFrame(hf.iloc[:,:11])
# y = hf['DEATH_EVENT']
# x = x.to_numpy()
# y = y.to_numpy()
# x.shape, y.shape
# x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2, random_state=1)

# def cal_score(x_train, y_train, x_test, y_test, indices):
#     lr = linear_model.LogisticRegression()
#     print(indices, x_train.shape)
#     lr.fit(x_train[:, indices], y_train)
#     y_pred = lr.predict(x_test[:, indices])
#     score = accuracy_score(y_test, y_pred)
#     return score

# from itertools import combinations

# score_list = []
# combin_list = []
# best_score_list = []

# for dim in range(1, x.shape[1]+1):
#     score_list = []
#     combin_list = []
    
#     all_dim = tuple(range(x.shape[1]))
    
#     for c in combinations(all_dim, r=dim):
#         score = cal_score(x_train, y_train, x_test, y_test, c)
#         score_list.append(score)
#         combin_list.append(c)
        
#     best_loc = np.argmax(score_list)
#     best_score = score_list[best_loc]
#     best_combin = combin_list[best_loc]
#     print(best_loc, best_combin, best_score)

# no = np.arange(1, len(best_score_list)+1)
# plt.plot(no, best_score_list, marker='o', markersize=6)
# 改座標範圍
# plt.ylim([0.6, 0.9])
# plt.xlim([0,11])
# plt.show()


# 用先前挑選出的4個欄位做預測
x = pd.DataFrame([hf['age'], hf['ejection_fraction'],
                  hf['serum_creatinine'], hf['high_blood_pressure']]).T

x_train, x_test, y_train, y_test = tts(x, y, test_size=0.4, random_state=1)

k = 7

knn = neighbors.KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)

pred = knn.predict(x_test)
print('\n用先前挑選出的4個特徵並使用KNN做預測')
print('accuracy =', accuracy_score(y_test, pred))

 # 繪製混淆矩陣
sebrn.set(font_scale = 1.8)
plt.figure(figsize=[10,6.6666667])
conf_matrix = (pd.crosstab(y_test, pred))
fx = sebrn.heatmap(conf_matrix, annot=True, cmap='Blues')

fx.set_title('\nConfusion Matrix - KNN\n')
fx.set_xlabel('Predicted Values')
fx.set_ylabel('Actual Values')

fx.xaxis.set_ticklabels(['0','1'])
fx.yaxis.set_ticklabels(['0','1'])

plt.show()

'''
(3) K-means
'''

from sklearn import cluster
print('='*50 + '\nK-means')

 # 4個特徵
x = pd.DataFrame([hf['age'], hf['ejection_fraction'],
                  hf['serum_creatinine'], hf['high_blood_pressure']]).T

k = 2
kmeans = cluster.KMeans(init='k-means++', n_clusters=k, random_state=12)
kmeans.fit(x)
print('4個特徵\n')
print('預測：\n',kmeans.labels_)
print('實際：\n',y)

pred = kmeans.labels_
print('accuracy =', accuracy_score(y, pred))

 # 5個特徵
x = pd.DataFrame([hf['age'],hf['creatinine_phosphokinase'],
                  hf['ejection_fraction'],hf['serum_creatinine'],
                  hf['high_blood_pressure']]).T

k = 2
kmeans = cluster.KMeans(init='k-means++', n_clusters=k, random_state=12)
kmeans.fit(x)
print('-'*40+'\n5個特徵\n')
print('預測：\n', kmeans.labels_)
print('實際：\n',y)

pred = kmeans.labels_
print('accuracy =', accuracy_score(y, pred))

 # 繪製混淆矩陣
sebrn.set(font_scale = 1.8)
plt.figure(figsize=[10,6.6666667])
conf_matrix = (pd.crosstab(y, pred))
fx = sebrn.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')

fx.set_title('\nConfusion Matrix - KMeans\n')
fx.set_xlabel('Predicted Values')
fx.set_ylabel('Actual Values')

fx.xaxis.set_ticklabels(['0','1'])
fx.yaxis.set_ticklabels(['0','1'])

plt.show()


