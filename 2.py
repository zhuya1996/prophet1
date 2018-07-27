from fbprophet import Prophet
import numpy as np
import pandas as pd

sales_df = pd.read_csv('baseline.csv')

sales_df['y_orig'] = sales_df['y'] # to save a copy of the original data..you'll see why shortly.
# log-transform y
sales_df['y'] = np.log(sales_df['y'])

model = Prophet() #instantiate Prophet
model.fit(sales_df); #fit the model with your dataframe

future_data = model.make_future_dataframe(periods=6, freq = 'm')

forecast_data = model.predict(future_data)
print(forecast_data)
forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

model.plot(forecast_data)
model.plot_components(forecast_data)

forecast_data_orig = forecast_data # make sure we save the original forecast data
forecast_data_orig['yhat'] = np.exp(forecast_data_orig['yhat'])
forecast_data_orig['yhat_lower'] = np.exp(forecast_data_orig['yhat_lower'])
forecast_data_orig['yhat_upper'] = np.exp(forecast_data_orig['yhat_upper'])
model.plot(forecast_data_orig)

sales_df['y_log']=sales_df['y'] #copy the log-transformed data to another column
sales_df['y']=sales_df['y_orig'] #copy the original data to 'y'

model.plot(forecast_data_orig)