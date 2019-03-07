import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from pandas.tseries.holiday import *
import datetime
from dateutil.relativedelta import relativedelta

import requests

df = web.DataReader(['RRSFS','INDPRO','HOUST','RPI','payems','UNRATE','M2SL','USREC','RECPROUSM156N'], 'fred', datetime.datetime(1980, 1, 2), datetime.datetime(2019, 3, 6))

df['retailyoy']=df['RRSFS']/df['RRSFS'].shift(12)-1
df['indyoy']=df['INDPRO']/df['INDPRO'].shift(12)-1
df['houstyoy']=df['HOUST']/df['HOUST'].shift(12)-1
df['rpiyoy']=df['RPI']/df['RPI'].shift(12)-1
df['empyoy']=df['payems']/df['payems'].shift(12)-1
df['UEma']=df['UNRATE']-df['UNRATE'].rolling(12).mean()
df['m2yoy']=df['M2SL']/df['M2SL'].shift(12)-1

cal = get_calendar('USFederalHolidayCalendar')  # Create calendar instance
cal.rules.pop(7)                                # Remove Veteran's Day rule
cal.rules.pop(6)                                # Remove Columbus Day rule
tradingCal = HolidayCalendarFactory('TradingCalendar', cal, GoodFriday)

df.index = df.index + pd.offsets.CustomBusinessMonthEnd(calendar=tradingCal())

df_daily = web.DataReader('^ixic', 'yahoo', datetime.datetime(1980, 1, 2), datetime.datetime(2019, 3, 6))
df_daily['200ma']=df_daily['Close']-df_daily['Close'].rolling(200).mean()
df1 = df_daily.resample('M').agg({'Open':'first', 'High':'max', 'Low': 'min', 'Close':'last'})
df1['month10ma']=df1['Close'].rolling(10).mean()
df1 = df1[:-1]
df1.index = df1.index + pd.offsets.CustomBusinessMonthBegin(calendar=tradingCal())-pd.offsets.CustomBusinessMonthEnd(calendar=tradingCal())

print(df1)

df = pd.concat([df_daily,df, df1],axis=1)
df = df.fillna(method='ffill')

df['retsales']=np.where(df['retailyoy']<=0, 1, 0)
df['indsales']=np.where(df['indyoy']<=0, 1, 0)
df['houstart']=np.where(df['houstyoy']<=-0.1, 1, 0)
df['personincome']=np.where(df['rpiyoy']<=-0, 1, 0)
df['jobs']=np.where(df['empyoy']<=-0, 1, 0)
df['emprate']=np.where(df['UEma']>=0, 1, 0)
df['recessionind']=df['retsales']+df['indsales']+df['houstart']+df['personincome']+df['jobs']+df['emprate']





df.to_csv('macro.csv')
