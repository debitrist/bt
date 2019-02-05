import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt


import time

start_time = time.time()

symbol = ['^vix','^gspc']
symbol.sort()
bars = web.DataReader(symbol, 'yahoo', datetime.datetime(1990, 1, 1), datetime.datetime(2019, 1, 29))

print("--- %s seconds ---" % (time.time() - start_time))
histreturn = bars['Close']/bars['Close'].shift()-1
rollma = bars['Close'].rolling(200).mean()
zrollma = (bars['Close']-rollma)/bars['Close'].rolling(200).std()
histreturn.columns = [['HistReturn']*len(histreturn.columns),histreturn.columns]
rollma.columns = [['RollMA']*len(rollma.columns),rollma.columns]
zrollma.columns = [['ZRollMA']*len(zrollma.columns),zrollma.columns]
bars = pd.concat([bars,histreturn,rollma,zrollma],axis=1)
for i in range(1,6):
    nreturn = (bars['Close'].shift(-i) / bars['Open'].shift(-1)) - 1 ##forward returns
    nreturn.columns = [['nreturn'+str(i)] * len(nreturn.columns), nreturn.columns]
    ndrawdown = (bars['Low'].shift(-1)[::-1].rolling(i).min()/bars['Open'].shift(-1)) - 1 ##forward drawdown
    ndrawdown.columns = [['ndrawdown' + str(i)] * len(ndrawdown.columns), ndrawdown.columns]
    bars = pd.concat([bars, nreturn,ndrawdown], axis=1)

print("--- %s seconds ---" % (time.time() - start_time))

##ATR
high_low = np.array(bars['High']-bars['Low'])
high_yc = abs(bars['High']-bars['Close'].shift())
low_yc = abs(bars['Low']-bars['Close'].shift())

tr = np.dstack((high_low, high_yc, low_yc))
tr1 = pd.DataFrame(tr.max(2), index=bars.index, columns=pd.MultiIndex.from_product([['TR'],symbol],names=['Attributes','Symbols']))
bars=pd.concat([bars,tr1],axis=1)
print("--- %s seconds ---" % (time.time() - start_time))

trclose = bars['TR']/bars['Close']
atrclose = (trclose).rolling(20).mean()
zatrclose = (trclose-atrclose)/trclose.rolling(20).std()
trclose.columns = [['TRClose']*len(trclose.columns),trclose.columns]
atrclose.columns = [['ATRClose']*len(atrclose.columns),atrclose.columns]
zatrclose.columns = [['ZATRClose']*len(zatrclose.columns),zatrclose.columns]
bars = pd.concat([bars,trclose,atrclose,zatrclose],axis=1)
print("--- %s seconds ---" % (time.time() - start_time))



for ticker in symbol:
    d = {ticker:{i: pd.DataFrame() for i in range(1,6)} for ticker in symbol}
    print(ticker)
    for i in range(1,6):
        nretstr = 'nreturn'+str(i)
        nddstr = 'ndrawdown'+str(i)
        stats = d[ticker][i]
        vix = bars['ATRClose', ticker].quantile(0.9)
        basedown1 = bars[bars['HistReturn', ticker] < 0]
        belowma = bars[bars['ZRollMA', ticker] < 0][bars['HistReturn', ticker] < 0]
        abovema = bars[bars['ZRollMA', ticker] > 0][bars['HistReturn', ticker] < 0]
        highervol = bars[bars['ZATRClose', ticker] > 0][bars['HistReturn', ticker] < 0]
        lowervol = bars[bars['ZATRClose', ticker] < 0][bars['HistReturn', ticker] < 0]
        highvix = bars[bars['ATRClose', ticker] > vix][bars['HistReturn', ticker] < 0]
        lowvix = bars[bars['ATRClose', ticker] < vix][bars['HistReturn', ticker] < 0]
        belowmahighvol = belowma[belowma['ZATRClose', ticker] > 2]
        belowmalowvol = belowma[belowma['ZATRClose', ticker] < -1]
        abovemahighvol = abovema[abovema['ZATRClose', ticker] > 2]
        abovemalowvol = abovema[abovema['ZATRClose', ticker] < -1]
        abovemahighvix = abovema[abovema['ATRClose', ticker]>vix]
        belowmahighvix = belowma[belowma['ATRClose',ticker]>vix]
        belowmahighvolhighvix = belowmahighvol[belowmahighvol['ATRClose',ticker] > vix]
        abovemahighvolhighvix = abovemahighvol[abovemahighvol['ATRClose',ticker] > vix]
        def label(stat,name):
            stats[str(name)+str(ticker)+str(i)]=stat[nretstr,ticker].describe()
            stats[str(name)+'d'+str(ticker)+str(i)]=stat[nddstr,ticker].describe()

        stats['baseline'+str(ticker)+str(i)] = bars[nretstr,ticker].describe()
        label(basedown1,'basedown1')
        label(highvix,'highvix')
        label(lowvix,'lowvix')
        label(belowmahighvol, 'bmahvol')
        label(belowmahighvolhighvix, 'bmahvolhvix')
        label(belowmalowvol,'bmalvol')
        label(abovemahighvol, 'amahvol')
        label(abovemahighvolhighvix,'amahvolhvix')
        label(abovemalowvol,'amalvol')
        label(highervol,'highervol')
        label(lowervol,'lowervol')
        label(abovemahighvix, 'amahvix')
        label(belowmahighvix, 'bmahvix')
        stats['abovema'+str(ticker)+str(i)]=abovema[nretstr,ticker].describe()
        stats['belowma'+str(ticker)+str(i)]= belowma[nretstr,ticker].describe()


    if ticker == '^gspc': break

summstats = pd.DataFrame([d['^gspc'][1].loc['count'].values,d['^gspc'][1].loc['mean'].values, d['^gspc'][2].loc['mean'].values, d['^gspc'][3].loc['mean'].values,d['^gspc'][4].loc['mean'].values,d['^gspc'][5].loc['mean'].values], index=["Days", "D1", "D2","D3","D4","D5"], columns=d['^gspc'][1].columns)



# list of dataframes
writer = pd.ExcelWriter('outputsheet4.xlsx')
row = 0
for i in range(1,6):
    d['^gspc'][i].to_excel(writer, sheet_name='stats', startrow=row, startcol=0)
    row = row + len(d['^gspc'][i].index) + 2 + 1

summstats.to_excel(writer, sheet_name='stats', startrow=0, startcol=len(d['^gspc'][1].columns)+2)

abovemahighvix.to_excel(writer, 'highvol')
bars.to_excel(writer, 'overall')
writer.save()


#print(bars[bars['ZRollMA','^gspc']<0][bars['ZATRClose', ticker] > 2]['nreturn','^gspc'].describe())

## PLOT

N = 5
fig, ax = plt.subplots()

ind = np.arange(N)    # the x locations for the groups
width = 0.1       # the width of the bars

benchmarkMeans = [d['^gspc'][i]['baseline^gspc'+str(i)].loc['mean'] for i in range(1,6)]
benchmarkStd = [d['^gspc'][i]['baseline^gspc'+str(i)].loc['std'] for i in range(1,6)]
p1 = ax.bar(ind, benchmarkMeans, width, color='r', bottom=0)

highvixMeans = [d['^gspc'][i]['highvix^gspc'+str(i)].loc['mean'] for i in range(1,6)]
highvixStd = [d['^gspc'][i]['highvix^gspc'+str(i)].loc['std'] for i in range(1,6)]
p2 = ax.bar(ind + width, highvixMeans, width, color='y', bottom=0)

amaMeans = [d['^gspc'][i]['amahvol^gspc'+str(i)].loc['mean'] for i in range(1,6)]
p3 = ax.bar(ind + width*2, amaMeans, width, color='b', bottom=0)

bmaMeans = [d['^gspc'][i]['bmahvol^gspc'+str(i)].loc['mean'] for i in range(1,6)]
p4 = ax.bar(ind + width*3, bmaMeans, width, color='b', bottom=0)




ax.set_title('Baseline vs filtered, n day returns')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('1D', '2D', '3D', '4D', '5D'))

ax.legend((p1[0], p2[0], p3[0], p4[0]), ('Benchmark', 'highvix', 'amahvol', 'bmahvol'))
ax.autoscale_view()

plt.show()

