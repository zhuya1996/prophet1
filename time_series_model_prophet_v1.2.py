# -*- coding: utf-8 -*-

"""
-----------------------------------------------
# @Time    : 2018/5/17 13:29
# @Author  : Dong.Wang
# @Email   :Dong.Wang@zkh360.com
# @File    : data_discover.py
# @ProjectName: Inventory_Optimization
------------------------------------------------

# @Brief:

"""
import pandas as pd
import numpy as np
import math
import time
import datetime as dt
from fbprophet import Prophet
import matplotlib.pyplot as plt
from data_base.db_mysql import MysqlEngine
from data_base.db_mysql import dbMySQL
from sklearn.metrics import mean_squared_error, r2_score
import calendar

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['figure.figsize'] = (12, 8)


def getLastDayOfLastMonth():
    d = dt.datetime.now()
    year = d.year
    month = d.month
    if month == 1:
        month = 12
        year -= 1
    else:
        month -= 1
    days = calendar.monthrange(year, month)[1]
    return dt.datetime(year, month, days)


def query_train_data_month(SkuNo):
    sku_quantity_sql = """SELECT
                                Years,
                                MonthOfYear,
                                WeekthOfYear,
                                OrderDate,
                                ROUND(DailyQuantity, 2) AS DailyQuantity
                            FROM
                                ai_sku_daily_quantity
                            WHERE
                                SkuNo = '{0}'
                            AND Years >= 2013
                            ORDER BY
                                Years ASC,
                                OrderDate ASC""".format(SkuNo)

    mysql_enngine = MysqlEngine()
    query_daily_result = mysql_enngine.query_db(sku_quantity_sql,index_col='OrderDate')
    starttime=dt.datetime.strptime('2013-01-01', '%Y-%m-%d')
    endtime=getLastDayOfLastMonth()
    date_range=pd.date_range(starttime, endtime)
    f_week=lambda x:int(time.strftime("%W",x.timetuple()))+1
    f_month=lambda x: int(time.strftime("%m", x.timetuple()))
    f_year=lambda x: int(time.strftime("%Y", x.timetuple()))

    full_week=list(map(f_week,date_range))
    full_month=list(map(f_month,date_range))
    full_year=list(map(f_year, date_range))
    df_structure=pd.DataFrame(data={'SkuNo':[SkuNo]*len(date_range),'WeekthOfYear':full_week,'MonthOfYear':full_month,'Years':full_year},index=date_range)
    df_structure=df_structure.join(query_daily_result,rsuffix='_new')
    df_structure.reset_index(inplace=True)
    df_structure.rename(columns={'index': 'OrderDate'}, inplace=True)
    month_result=df_structure.groupby(by=['Years', 'MonthOfYear']).agg({'WeekthOfYear': np.max, 'OrderDate': np.max, 'DailyQuantity': np.sum})
    month_result.reset_index(inplace=True)
    month_result.rename(columns={'DailyQuantity': 'MonthQuantity'}, inplace=True)

    month_result_orign=month_result
    month_result=month_result[month_result.MonthQuantity > 0]


    # 缺失值填充、异常值处理
    sku_month_quantity = anomaly_detection_month(month_result)

    return sku_month_quantity,month_result_orign


def query_train_data_week(SkuNo):
    sku_quantity_sql = """SELECT
                                Years,
                                MonthOfYear,
                                WeekthOfYear,
                                OrderDate,
                                ROUND(DailyQuantity, 2) AS DailyQuantity
                            FROM
                                ai_sku_daily_quantity
                            WHERE
                                SkuNo = '{0}'
                            AND Years >= 2013
                            ORDER BY
                                Years ASC,
                                OrderDate ASC""".format(SkuNo)

    mysql_enngine = MysqlEngine()
    query_daily_result = mysql_enngine.query_db(sku_quantity_sql,index_col='OrderDate')
    starttime=dt.datetime.strptime('2013-01-01', '%Y-%m-%d')
    endtime=dt.datetime.today() - dt.timedelta(days=dt.datetime.now().weekday() + 1)
    date_range=pd.date_range(starttime, endtime)
    f_week=lambda x:int(time.strftime("%W",x.timetuple()))+1
    f_month=lambda x: int(time.strftime("%m", x.timetuple()))
    f_year=lambda x: int(time.strftime("%Y", x.timetuple()))

    full_week=list(map(f_week,date_range))
    full_month=list(map(f_month,date_range))
    full_year=list(map(f_year, date_range))
    df_structure=pd.DataFrame(data={'SkuNo':[SkuNo]*len(date_range),'WeekthOfYear':full_week,'MonthOfYear':full_month,'Years':full_year},index=date_range)
    df_structure=df_structure.join(query_daily_result,rsuffix='_new')
    df_structure.reset_index(inplace=True)
    df_structure.rename(columns={'index': 'OrderDate'}, inplace=True)
    week_result=df_structure.groupby(by=['Years', 'WeekthOfYear']).agg({'MonthOfYear': np.max, 'OrderDate': np.max, 'DailyQuantity': np.sum})
    week_result.reset_index(inplace=True)
    week_result.rename(columns={'DailyQuantity': 'WeekQuantity'}, inplace=True)
    # 缺失值填充、异常值处理
    week_result_orign=week_result
    week_result=week_result[week_result.WeekQuantity > 0]
    week_result.reset_index(inplace=True)
    sku_weeks_quantity = anomaly_detection_week(week_result)

    return sku_weeks_quantity,week_result_orign


def update_table(AiPredictQuantity,AiPredictStep,SkuNo):
    sql="""UPDATE ai_sku_stock_optimize SET AiPredictQuantity={0},AiPredictStep={1} WHERE SkuNo='{2}' """
    db.cur.execute(sql.format(AiPredictQuantity,AiPredictStep,SkuNo))

def anomaly_detection_week(df):
    df.WeekQuantity = df.WeekQuantity.replace(0, 0.01)
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

def anomaly_detection_month(df):
    df.MonthQuantity = df.MonthQuantity.replace(0, 0.01)
    q1 = np.percentile(df.MonthQuantity[df.MonthQuantity > 0.1].tolist(), 25)
    q3 = np.percentile(df.MonthQuantity[df.MonthQuantity > 0.1].tolist(), 75)

    iqr = q3 - q1
    max_line = q3 + 3 * iqr
    min_line = q1 - 3 * iqr
    df.MonthQuantity[df.MonthQuantity > max_line] = np.nan
    df.MonthQuantity.fillna(max_line, inplace=True)
    df.MonthQuantity[df.MonthQuantity < min_line] = np.nan
    df.MonthQuantity.fillna(min_line, inplace=True)
    return df

def model_train_week(SkuNo,numPredictStep):
    sales_df_orign,week_result_orign=query_train_data_week(SkuNo)
    week_result_orign=week_result_orign.rename(columns={'OrderDate': 'ds', 'WeekQuantity': 'y'})
    sales_df_orign = sales_df_orign.rename(columns={'OrderDate': 'ds', 'WeekQuantity': 'y'})
    sales_df_orign.y=np.log(sales_df_orign.y)
    model = Prophet()
    # model.add_seasonality(name='monthly', period=30, fourier_order=5,prior_scale=2., mode='additive')
    sales_df=sales_df_orign.head(len(sales_df_orign)-numPredictStep)
    model.fit(sales_df)
    future=model.make_future_dataframe(periods=numPredictStep, freq='W')
    forecast = model.predict(future)
    sales_df_orign.y=np.exp(sales_df_orign.y)
    forecast.yhat=np.exp(forecast.yhat)
    forecast.yhat_lower=np.exp(forecast.yhat_lower)
    forecast.yhat_upper=np.exp(forecast.yhat_upper)
    # predict_value=forecast.yhat[-numPredictStep:].apply(lambda  x :math.ceil(x))
    # print(predict_value)
    # total_predict_value=predict_value.sum()
    # update_table(total_predict_value,numPredictStep,SkuNo)
    train_mse=mean_squared_error(sales_df.y,forecast.yhat[:-numPredictStep])
    predict_mse=mean_squared_error(sales_df_orign.tail(numPredictStep).y, forecast.yhat[-numPredictStep:])
    # r2=r2_score(np.exp(sales_df.y),forecast.yhat[:-numPredictStep])
    fig=plt.figure(figsize=(12,6))
    ax1=fig.add_subplot(1,1,1)
    ax1.plot(forecast.ds[:-numPredictStep+1],forecast.yhat[:-numPredictStep+1],color='blue',label='拟合值')
    ax1.plot(forecast.ds[-numPredictStep:], forecast.yhat[-numPredictStep:], color='red', linestyle=':', label='预测值')
    ax1.plot(week_result_orign.ds,week_result_orign.y,color='green',label='真实值')
    plt.legend(['拟合值','预测值', '真实值'])
    plt.title('Prophet Model,Predict SKU:{0}\n TrainMSE:{1},TestMSE:{2}'.format(SkuNo,round(train_mse,2),round(predict_mse,2)))
    fig.savefig('./ModelImage/Prophet_Week_{0}.png'.format(SkuNo))
    # plt.show()

def model_train_month(SkuNo,numPredictStep):
    sales_df_orign,month_result_orign=query_train_data_month(SkuNo)
    month_result_orign=month_result_orign.rename(columns={'OrderDate': 'ds', 'MonthQuantity': 'y'})
    sales_df_orign = sales_df_orign.rename(columns={'OrderDate': 'ds', 'MonthQuantity': 'y'})
    sales_df_orign.y=np.log(sales_df_orign.y)
    model = Prophet()
    # model.add_seasonality(name='monthly', period=30, fourier_order=5,prior_scale=2., mode='additive')
    sales_df=sales_df_orign.head(len(sales_df_orign)-numPredictStep)
    model.fit(sales_df)
    future=model.make_future_dataframe(periods=numPredictStep, freq='M')
    forecast = model.predict(future)
    sales_df_orign.y=np.exp(sales_df_orign.y)
    forecast.yhat=np.exp(forecast.yhat)
    forecast.yhat_lower=np.exp(forecast.yhat_lower)
    forecast.yhat_upper=np.exp(forecast.yhat_upper)
    # predict_value=forecast.yhat[-numPredictStep:].apply(lambda  x :math.ceil(x))
    # print(predict_value)
    # total_predict_value=predict_value.sum()
    # update_table(total_predict_value,numPredictStep,SkuNo)
    train_mse=mean_squared_error(sales_df.y,forecast.yhat[:-numPredictStep])
    predict_mse=mean_squared_error(sales_df_orign.tail(numPredictStep).y, forecast.yhat[-numPredictStep:])
    # r2=r2_score(np.exp(sales_df.y),forecast.yhat[:-numPredictStep])
    fig=plt.figure(figsize=(12,6))
    ax1=fig.add_subplot(1,1,1)
    ax1.plot(forecast.ds[:-numPredictStep+1],forecast.yhat[:-numPredictStep+1],color='blue',label='拟合值')
    ax1.plot(forecast.ds[-numPredictStep:], forecast.yhat[-numPredictStep:], color='red', linestyle=':', label='预测值')
    ax1.plot(month_result_orign.ds,month_result_orign.y,color='green',label='真实值')
    plt.legend(['拟合值','预测值', '真实值'])
    plt.title('Prophet Model,Predict SKU:{0}\n TrainMSE:{1},TestMSE:{2}'.format(SkuNo,round(train_mse,2),round(predict_mse,2)))
    fig.savefig('./ModelImage/Prophet_Month_{0}.png'.format(SkuNo))
    # plt.show()


if __name__ == "__main__":
    select_sku_sql="""SELECT
                    t1.SkuNo,
                    t1.NumOfExistSalesIn90Day,
                    CEIL(
                        90 / t1.NumOfExistSalesIn90Day / 7
                    ) PerWeekExistSale,
                    t2.NumOfExistSalesWeeks
                FROM
                    (
                        SELECT
                            SkuNo,
                            COUNT(1) AS NumOfExistSalesIn90Day
                        FROM
                            ai_sku_weeks_quantity
                        WHERE
                            OrderDate >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)
                        GROUP BY
                            SkuNo
                        HAVING
                            NumOfExistSalesIn90Day >= 12
                    ) t1
                INNER JOIN (
                    SELECT
                        SkuNo,
                        COUNT(1) AS NumOfExistSalesWeeks
                    FROM
                        ai_sku_weeks_quantity
                    GROUP BY
                        SkuNo
                    HAVING
                        NumOfExistSalesWeeks >= 120
                ) t2 ON t1.SkuNo = t2.SkuNo
                ORDER BY RAND()
                LIMIT 25"""

    db=dbMySQL()
    cnt=db.cur.execute(select_sku_sql)
    print('共要预测{0}个SKU'.format(cnt))

    for i,sku_info in enumerate(db.cur.fetchall()):
        print('开始预测第{0}个SKU'.format(i+1))
        model_train_week(SkuNo =sku_info['SkuNo'],numPredictStep=20)
        model_train_month(SkuNo=sku_info['SkuNo'], numPredictStep=5)
        db.iCommit()
    db.closeDB()