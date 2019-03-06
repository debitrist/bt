import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import requests

df = web.DataReader(['RRSFS','INDPRO','HOUST','RPI','payems','M2SL','USREC','RECPROUSM156N'], 'fred', datetime.datetime(1980, 1, 2), datetime.datetime(2019, 3, 6))

df['retailyoy']=df['RRSFS']/df['RRSFS'].shift(12)-1
df['indyoy']=df['INDPRO']/df['INDPRO'].shift(12)-1
df['houstyoy']=df['HOUST']/df['HOUST'].shift(12)-1
df['rpiyoy']=df['RPI']/df['RPI'].shift(12)-1
df['empyoy']=df['payems']/df['payems'].shift(12)-1
df['m2yoy']=df['M2SL']/df['M2SL'].shift(12)-1
df.index = df.index + pd.offsets.BMonthEnd(1)

df_daily = web.DataReader('^gspc', 'yahoo', datetime.datetime(1980, 1, 2), datetime.datetime(2019, 3, 6))

df = pd.concat([df_daily,df],axis=1)
df=df.fillna(method='ffill')




df.to_csv('macro.csv')
