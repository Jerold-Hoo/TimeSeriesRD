# -*- coding: utf-8 -*-

# @File       : ARIMA.py
# @Date       : 2019-10-27
# @Author     : Jerold
# @Description: use Arima

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

data = pd.read_csv(r'KWH_Data\train.csv')
data = data[['time','KWH']]
data.index = pd.to_datetime(data['time'])
data = data['KWH']

def draw_SMAtrend(timeseries, size):
    rol_mean = timeseries.rolling(window=size).mean()
    rol_std = timeseries.rolling(window=size).std()

    #print(rol_mean.head())
    #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y %H'))
    #plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

    plt.plot(timeseries.index,timeseries.values)
    plt.plot(rol_mean.index, rol_mean.values)

    #plt.gcf().autofmt_xdate()
    plt.show()

draw_SMAtrend(data, 24)