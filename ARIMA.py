# -*- coding: utf-8 -*-

# @File       : ARIMA.py
# @Date       : 2019-10-27
# @Author     : Jerold
# @Description: use Arima

import pandas as pd

data = pd.read_csv(r'KWH_Data\train.csv')
data = data[['time','KWH']]
data.index = pd.to_datetime(data['time'])
data = data['KWH']

def draw_SMAtrend(timeseries, size):
    rol_mean = timeseries.rolling(window=size).mean()
    rol_std = timeseries.rolling(window=size).std()
    print(rol_mean.head(50))
    print(rol_std.head(50))

draw_SMAtrend(data, 10)