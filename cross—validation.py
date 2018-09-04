import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model

data = pd.read_csv('.\CCPP\ccpp.csv')

#读取前五行数据，如果是最后五行，用data.tail()
data.head()
#我们看看数据的维度：
data.shape
　#现在我们开始准备样本特征X，我们用AT， V，AP和RH这4个列作为样本特征。

X = data[['AT', 'V', 'AP', 'RH']]
X.head()

#接着我们准备样本输出y， 我们用PE作为样本输出。

y = data[['PE']]
y.head()

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

　查看下训练集和测试集的维度：

print X_train.shape
print y_train.shape
print X_test.shape
print y_test.shape

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)

print linreg.intercept_
print linreg.coef_

复制代码
#模型拟合测试集
y_pred = linreg.predict(X_test)
from sklearn import metrics
# 用scikit-learn计算MSE
print "MSE:",metrics.mean_squared_error(y_test, y_pred)
# 用scikit-learn计算RMSE
print "RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred))



fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()





#https://www.cnblogs.com/pinard/p/6016029.html
