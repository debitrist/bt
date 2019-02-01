import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as web

import time

start_time = time.time()

symbol = ('spy', '^vix', 'aapl', 'smar')
bars = web.DataReader(symbol, 'yahoo', datetime.datetime(2010, 1, 1), datetime.datetime(2019, 1, 29))
symbol = ['spy', '^vix', 'aapl', 'smar']

print("--- %s seconds ---" % (time.time() - start_time))

test = bars['Close'].rolling(20).mean()
print(test)
test.columns = [['RollMA']*len(test.columns),test.columns]

bars = pd.concat([bars,test],axis=1)
print("--- %s seconds ---" % (time.time() - start_time))

high_low = bars['High']-bars['Low']
high_yc = abs(bars['High']-bars['Close'].shift())
low_yc = abs(bars['Low']-bars['Close'].shift())


bars.columns = bars.columns.swaplevel(0, 1)
bars.sort_index(axis=1, level=0, inplace=True)


print(bars['spy'].columns)



bars.to_csv('bars2.csv')
