# -*- coding: utf-8 -*-

"""
-----------------------------------------------
# @Time    : 2018/7/17 13:29
# @Author  : mengqi.zhu
# @Email   : mengqi.zhu@zkh360.com
# @File    :
# @ProjectName: Inventory_Optimization
------------------------------------------------

# @Brief:

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataBase.db_mysql import MysqlEngine
from sklearn import preprocessing
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers.core import Dense, Activation

# plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.rc('font', family='DejaVu Sans')
plt.rc('font', **{'sans-serif': 'Arial', 'family': 'sans-serif'})
plt.figure(figsize=(15, 5))


def query_train_data(SkuNo):
    sku_quantity_sql = """SELECT
                          1stDayOfMonth,
                          MonthQuantity,
                          MonthOrderTimesNum,
                          MonthCustomerNum,
                          MonthAvgOrderPrice
                        FROM
                            ai_sku_month_feature
                        WHERE
                            SkuNo = '{0}'
                        ORDER BY
                            1stDayOfMonth asc """.format(SkuNo)

    mysql_enngine = MysqlEngine()
    # mysql_enngine.set_con()
    query_original_result = mysql_enngine.query_db(sku_quantity_sql)
    # query_result.fillna(0)
    process_result = query_original_result.copy()

    Pre1_Q = process_result.MonthQuantity.shift(1)
    Pre2_Q = process_result.MonthQuantity.shift(2)
    # Pre3_Q = process_result.MonthQuantity.shift(3)

    Pre1_OTimes = process_result.MonthOrderTimesNum.shift(1)
    Pre2_OTimes = process_result.MonthOrderTimesNum.shift(2)
    # Pre3_OTimes = process_result.MonthOrderTimesNum.shift(3)

    Pre1_CusNum = process_result.MonthCustomerNum.shift(1)
    Pre2_CusNum = process_result.MonthCustomerNum.shift(2)
    # Pre3_CusNum = process_result.MonthCustomerNum.shift(3)

    # Pre1_Price = process_result.MonthAvgOrderPrice.shift(1)
    # Pre2_Price = process_result.MonthAvgOrderPrice.shift(2)
    # Pre3_Price = process_result.MonthAvgOrderPrice.shift(3)

    # 将后一个月的销量加入到当前月特征集，NextMonthQuantity作为预测中的Y值
    process_result['NextMonthQuantity'] = process_result.MonthQuantity.shift(-1)

    data = pd.concat([process_result['NextMonthQuantity'],
                      process_result['MonthQuantity'],
                      process_result['MonthOrderTimesNum'],
                      process_result['MonthCustomerNum'],
                      Pre1_Q, Pre2_Q,
                      Pre1_OTimes, Pre2_OTimes,
                      Pre1_CusNum, Pre2_CusNum
                      ], axis=1)

    data.columns = ['NextMonthQuantity', 'MonthQuantity',
                    'MonthOrderTimesNum', 'MonthCustomerNum',
                    'Pre1_Q', 'Pre2_Q',
                    'Pre1_OTimes', 'Pre2_OTimes',
                    'Pre1_CusNum', 'Pre2_CusNum']
    print data.head(4)
    print data.tail(4)
    data = data.dropna()
    # print data.tail(6)
    print data.head(4)
    print data.tail(4)

    # 缺失值填充、异常值处理
    process_result = anomaly_detection(data)
    # print(query_original_result)

    return process_result, query_original_result


def anomaly_detection(df):
    df.MonthQuantity = df.MonthQuantity.fillna(0)
    df.MonthQuantity = df.MonthQuantity.replace(0, 0.01)

    df.MonthOrderTimesNum = df.MonthOrderTimesNum.fillna(0)
    df.MonthOrderTimesNum = df.MonthOrderTimesNum.replace(0, 0.1)

    df.MonthCustomerNum = df.MonthCustomerNum.fillna(0)
    df.MonthCustomerNum = df.MonthCustomerNum.replace(0, 0.1)

    # df.MonthAvgOrderPrice = df.MonthAvgOrderPrice.fillna(0)
    # df.MonthAvgOrderPrice = df.MonthAvgOrderPrice.replace(0, 0.1)
    # handle_data_by_box_plot(df, df.MonthAvgOrderPrice, 'MonthAvgOrderPrice')

    handle_data_by_box_plot(df, df.MonthQuantity, 'MonthQuantity')
    handle_data_by_box_plot(df, df.NextMonthQuantity, 'NextMonthQuantity')

    # # 销量异常值处理
    # q1_quantity = np.percentile(df.MonthQuantity[df.MonthQuantity > 0].tolist(), 25)
    # q3_quantity = np.percentile(df.MonthQuantity[df.MonthQuantity > 0].tolist(), 75)
    # iqr = q3_quantity - q1_quantity
    # max_line = q3_quantity + 3 * iqr
    # min_line = q1_quantity - 3 * iqr
    # df.MonthQuantity[df.MonthQuantity > max_line] = np.nan
    # df.MonthQuantity.fillna(max_line, inplace=True)
    # df.MonthQuantity[df.MonthQuantity < min_line] = np.nan
    # df.MonthQuantity.fillna(min_line, inplace=True)
    #
    return df


def handle_data_by_box_plot(df, df_field, field_str):
    q1 = np.percentile(df_field[df_field > 0].tolist(), 25)
    q3 = np.percentile(df_field[df_field > 0].tolist(), 75)
    iqr = q3 - q1
    max_line = q3 + 3 * iqr
    min_line = q1 - 3 * iqr
    df_field[df_field > max_line] = np.nan
    df_field.fillna(max_line, inplace=True)
    df_field[df_field < min_line] = np.nan
    df_field.fillna(min_line, inplace=True)
    df[field_str] = df_field


if __name__ == "__main__":
    # AC2400, AM2291  AE6949(148 三倍内)
    # pipe = joblib.load('AE2235.pkl')
    # print(pipe.forecast(1));
    #
    # predictNextDayStock('AE2235.pkl')
    SkuNo = "AM2291"
    process_result, query_original_result = query_train_data(SkuNo)

    # 最后一行X值只用于预测Y，先保留在变量中。
    lastInputsX = process_result.tail(1)
    print("-----------: ")
    print(lastInputsX)

    lastInputsX.drop('NextMonthQuantity', inplace=True, axis=1)
    print("-----drop------: ")
    print(lastInputsX)
    # lastInputsX = lastInputsX.apply(np.log)
    lastInputsX = np.array(lastInputsX).reshape((len(lastInputsX), 9))

    scaler_x = preprocessing.MinMaxScaler(feature_range=(0, 1))
    lastInputsX = scaler_x.fit_transform(lastInputsX)
    lastInputsX = lastInputsX.reshape(lastInputsX.shape + (1,))

    sizeOfResult = len(process_result)
    # 去掉最后一月数据，保留其它数据用于训练、测试模型
    process_result.drop(process_result.index[[sizeOfResult - 1]], inplace=True)
    print("-----------process_result: ")
    print(process_result)

    y = process_result['NextMonthQuantity']
    print("-----------process_result2: ")
    print(process_result)

    x = process_result.drop('NextMonthQuantity', axis=1)
    # x = x.apply(np.log)

    x = np.array(x).reshape((len(x), 9))
    x = scaler_x.fit_transform(x)

    scaler_y = preprocessing.MinMaxScaler(feature_range=(0, 1))
    y = np.array(y).reshape((len(y), 1))
    # y = np.log(y)
    y = scaler_y.fit_transform(y)

    end = len(x) - 1
    learn_end = int(end * 0.9)
    x_train = x[0: learn_end - 1, ]
    x_test = x[learn_end: end - 1, ]
    y_train = y[1: learn_end]
    y_test = y[learn_end + 1: end]
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))

    x_last = x[end: end + 1, ]
    x_last = x_last.reshape(x_last.shape + (1,))
    x_all = x[0:]
    x_all = x_all.reshape(x_all.shape + (1,))

    print "Shape of x_train is ", x_train.shape
    print "---------------------------------"
    print "Shape of x_test is ", x_test.shape

    seed = 2016
    np.random.seed(seed)

    fit1 = Sequential()
    fit1.add(LSTM(output_dim=1,
                  activation='tanh', inner_activation='hard_sigmoid', input_shape=(9, 1)))
    fit1.add(Dense(output_dim=1, activation='linear'))
    fit1.compile(loss="mean_squared_error", optimizer="rmsprop")
    fit1.fit(x_train, y_train, batch_size=1, nb_epoch=20, shuffle=True)
    score_train = fit1.evaluate(x_train, y_train, batch_size=1)
    score_test = fit1.evaluate(x_test, y_test, batch_size=1)
    print "in train MSE = ", round(score_train, 4)
    print "in test MSE = ", round(score_test, 4)

    pred1 = fit1.predict(x_test)
    pred1 = scaler_y.inverse_transform(np.array(pred1).reshape((len(pred1), 1)))
    # pred1 = np.exp(pred1)

    print "----------query_original_result MonthQuantity----------"
    print(query_original_result["MonthQuantity"])

    print "----------pred1----------"
    print np.rint(pred1)

    print "----------fit1.predict(x_last)----------"
    pred_last = fit1.predict(lastInputsX)
    pred_last = scaler_y.inverse_transform(np.array(pred_last).reshape((len(pred_last), 1)))
    # pred_last = round(np.exp(pred_last),2)
    # pred_last = np.rint(np.exp(pred_last))
    pred_last = np.rint(pred_last)
    print pred_last

    print "----------fit1.predict(x_all)----------"
    pred_all = fit1.predict(x_all)
    pred_all = scaler_y.inverse_transform(np.array(pred_all).reshape((len(pred_all), 1)))


    pred_all_series = pd.Series(pred_all[:, 0])
    fillMonthQuantity = query_original_result["MonthQuantity"].head(4)
    print "----------fillMonthQuantity----------"

    print(fillMonthQuantity)
    print(pred_last)
    pred_all = fillMonthQuantity.append(pred_all_series, ignore_index=True)
    # pred_all = pred_all.append(pred_last, ignore_index=True)

    pred_all = np.array(pred_all)
    # pred_all = np.exp(pred_all)

    print "------------------"
    print np.rint(pred_all)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(pred_all, color='blue', label='Predict Values')
    ax1.plot(query_original_result["MonthQuantity"], color='green', label='Real Values')
    plt.legend(['Predict Values', 'Real Values'])
    plt.title("LSTM Model Predict Sku '{0}'       MSE:{1}".format(SkuNo, round(score_test, 3)))
    plt.show()
