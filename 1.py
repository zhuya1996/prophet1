#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt
import tushare as ts
de=ts.get_hist_data('002230',start='2016-01-25',end='2017-11-01')

df=pd.DataFrame()

df['y']=de['open']

df['ds']=list(de.index)
print(df['ds'])
# 定义模型
m = Prophet()

# 训练模型
m.fit(df)
print(df)
# 构建预测集
future = m.make_future_dataframe(periods=5)
print(future)
# print (future.tail())

# 进行预测
forecast = m.predict(future)
print(forecast)
m.plot(forecast)
plt.show()