# -*- coding: utf-8 -*-

"""
-----------------------------------------------
# @Time    : 2018/5/17 13:29
# @Author  : mengqi.zhu
# @Email   :mengqi.zhu@zkh360.com
# @File    :
# @ProjectName: Inventory_Optimization
------------------------------------------------

# @Brief:

"""
import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt
from data_base.db_mysql import MysqlEngine
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

plt.rcParams['figure.figsize'] = (20, 10)
plt.style.use('ggplot')


def query_train_data(SkuNo):
    sku_quantity_sql = """SELECT
                          SalesDate as ds,
                          ROUND(WeekQuantity, 2) as WeekQuantity
                        FROM
                            ai_sku_weeks_quantity_JiaoYou
                        WHERE
                            SkuNo = '{0}'
                            AND Years>=2013
                            AND WeekQuantity 
                        ORDER BY
                            Years ASC,
                            WeekthOfYear ASC""".format(SkuNo)

    mysql_enngine = MysqlEngine()
    mysql_enngine.set_con()
    query_result = mysql_enngine.query_db(sku_quantity_sql)
    # 缺失值填充、异常值处理
    sku_weeks_quantity = anomaly_detection(query_result)

    return sku_weeks_quantity


# def anomaly_detection(df):
#     df.WeekQuantity = df.WeekQuantity.fillna(0)
#     df.WeekQuantity = df.WeekQuantity.replace(0, 0.1)
#     q1 = np.percentile(df.WeekQuantity[df.WeekQuantity > 0.1].tolist(), 25)
#     q3 = np.percentile(df.WeekQuantity[df.WeekQuantity > 0.1].tolist(), 75)
#
#     iqr = q3 - q1
#     max_line = q3 + 3 * iqr
#     min_line = q1 - 3 * iqr
#     df.WeekQuantity[df.WeekQuantity > max_line] = np.nan
#     df.WeekQuantity.fillna(max_line, inplace=True)
#     df.WeekQuantity[df.WeekQuantity < min_line] = np.nan
#     df.WeekQuantity.fillna(min_line, inplace=True)
#     return df
def anomaly_detection(df):
    df.WeekQuantity = df.WeekQuantity.fillna(0)
    df.WeekQuantity = df.WeekQuantity.replace(0, 0.1)
    q1 = np.percentile(df.WeekQuantity[df.WeekQuantity > 0.1].tolist(), 25)
    q3 = np.percentile(df.WeekQuantity[df.WeekQuantity > 0.1].tolist(), 75)

    iqr = q3 - q1
    max_line = q3 + 3 * iqr
    min_line = q1 - 3 * iqr
    df.WeekQuantity[df.WeekQuantity > max_line] = np.nan
    df.WeekQuantity.fillna(max_line, inplace=True)
    df.WeekQuantity[df.WeekQuantity < min_line] = np.nan
    df.WeekQuantity.fillna(min_line, inplace=True)
    return df

if __name__ == "__main__":
    SkuNo = "AH0818"  #AE5955  AA9513
    numPredictStep = 4
    sales_df = query_train_data(SkuNo)

    dataSetSize = len(sales_df)
    # print "---------dataSetSize-----------"
    # print dataSetSize

    # if dataSetSize <= 100:
    #     plt.figure(figsize=(10, 5))
    # elif dataSetSize > 100 & dataSetSize <= 200:
    #     plt.figure(figsize=(18, 8))
    # elif dataSetSize > 200 & dataSetSize <= 300:
    #     plt.figure(figsize=(24, 8))
    # elif dataSetSize > 300 & dataSetSize <= 400:
    #     plt.figure(figsize=(30, 10))
    # else:
    #     plt.figure(figsize=(35, 12))

    # print("---sales_df.head(50)--")
    # print sales_df.head(20)
    # print sales_df.tail(20)

    df = sales_df.reset_index()

    df = df.rename(columns={'SalesDate': 'ds', 'WeekQuantity': 'y'})
    df.head()
    df.set_index('ds').y.plot()
    df['y'] = np.log(df['y'])
    df.tail()
    df.set_index('ds').y.plot()
    # model = Prophet()
    model = Prophet(weekly_seasonality=True)
    model.fit(df)

    future = model.make_future_dataframe(periods=numPredictStep, freq='w')
    # print("--------future.tail()------")
    # print future.tail()

    forecast = model.predict(future)

    print("--------forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()------")
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

    df.set_index('ds', inplace=True)
    forecast.set_index('ds', inplace=True)
    viz_df = sales_df.join(forecast[['yhat', 'yhat_lower', 'yhat_upper']], how='outer')

    metric_df = forecast[['yhat']].join(df.y).reset_index()
    metric_df.dropna(inplace=True)

    print("-----len(metric_df)------")

    print len(metric_df)

    print("--------r2_score------")
    print r2_score(metric_df.y, metric_df.yhat)
    print("--------mean_squared_error------")
    # MSE预测数据和原始数据对应点误差的平方和的均值
    print mean_squared_error(metric_df.y, metric_df.yhat)
    print("--------mean_absolute_error------")
    print mean_absolute_error(metric_df.y, metric_df.yhat)

    print("metric_df.yhat")
    metric_df.yhat

    viz_df['yhat_rescaled'] = np.exp(viz_df['yhat'])

    # sales_df.set_index('ds', inplace=True)
    sales_df.index = pd.to_datetime(sales_df.ds)  # make sure our index as a datetime object
    connect_date = sales_df.index[-2]  # select the 2nd to last date

    # print("-------after sales_df.index-----")
    # print(sales_df.index)

    mask = (forecast.index > connect_date)
    predict_df = forecast.loc[mask]

    # print("--------predict_df.head()----------")
    # print predict_df.head()
    # print predict_df.tail()
    #
    # print("--------sales_df.head()----------")
    # print sales_df.head()
    # print sales_df.tail()

    viz_df = sales_df.join(predict_df[['yhat', 'yhat_lower', 'yhat_upper']], how='outer')
    viz_df['yhat_scaled'] = np.exp(viz_df['yhat'])
    viz_df['yhat_scaled'] = np.exp(viz_df['yhat_lower'])
    viz_df['yhat_scaled'] = np.exp(viz_df['yhat_upper'])

    # print("--------sales_df.join(predict_df[['yhat', 'yhat_lower','yhat_upper']], how = 'outer')----------")
    # viz_df.head()
    # viz_df.tail()
    print("--------Predict Next " + str(numPredictStep) + " Weeks---------")
    print viz_df.tail(numPredictStep).yhat_scaled

    # plt.plot(metric_df.y[:-numPredictStep])
    # plt.plot(metric_df.yhat[:-numPredictStep])

    fig, ax1 = plt.subplots()
    ax1.plot(viz_df.WeekQuantity)
    ax1.plot(viz_df.yhat_scaled, color='black', linestyle=':')
    ax1.fill_between(viz_df.index, np.exp(viz_df['yhat_upper']), np.exp(viz_df['yhat_lower']), alpha=0.5,
                     color='darkgray')

    ax1.set_title(
        "Historical Sales Amount(Orange) vs Sales Forecast (Black)" + "\n SKU : " + SkuNo + "   Prediction Range: " + str(
            numPredictStep) + " weeks in future")
    ax1.set_ylabel('Sales Quantity')
    ax1.set_xlabel('Date')

    L = ax1.legend()  # get the legend
    L.get_texts()[0].set_text('Actual Sales')  # change the legend text for 1st plot
    L.get_texts()[1].set_text('Forecasted Sales')  # change the legend text for 2nd plot

    # from fbprophet.diagnostics import cross_validation
    #
    # df_cv = cross_validation(df, initial='80', period='10', horizon='35')
    # df_cv.head()
    # from fbprophet.diagnostics import performance_metrics
    #
    # df_p = performance_metrics(df_cv)
    # print("--------df_p.head()-------")
    # print df_p.head()

    plt.show()

    # import pickle
    #
    # save_model = pickle.dumps(model)
    # model2 = pickle.loads(save_model)
    # model2.plot(forecast);
