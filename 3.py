# Python
import pandas as pd
from fbprophet import Prophet

df = pd.read_csv('example_wp_log_peyton_manning.csv')
df.head()

playoffs = pd.DataFrame({
  'holiday': 'playoff',
  'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                        '2010-01-24', '2010-02-07', '2011-01-08',
                        '2013-01-12', '2014-01-12', '2014-01-19',
                        '2014-02-02', '2015-01-11', '2016-01-17',
                        '2016-01-24', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
superbowls = pd.DataFrame({
  'holiday': 'superbowl',
  'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
holidays = pd.concat((playoffs, superbowls))

print(holidays)
# Python
m = Prophet()
m.fit(df)

# Python
future = m.make_future_dataframe(periods=365)
print(future)


future.tail()

# Python
forecast = m.predict(future)
print(forecast)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# Python
fig1 = m.plot(forecast)
# Python
fig2 = m.plot_components(forecast)
