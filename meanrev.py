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
bars = web.DataReader(symbol, 'yahoo', datetime.datetime(2005, 1, 1), datetime.datetime(2019, 1, 29))

print("--- %s seconds ---" % (time.time() - start_time))
histreturn = bars['Close']/bars['Close'].shift()-1
rollma = bars['Close'].rolling(200).mean()
zrollma = (bars['Close']-rollma)/bars['Close'].rolling(200).std()
ibs = (bars['Close']-bars['Low'])/(bars['High']-bars['Low'])
histreturn.columns = [['HistReturn']*len(histreturn.columns),histreturn.columns]
rollma.columns = [['RollMA']*len(rollma.columns),rollma.columns]
zrollma.columns = [['ZRollMA']*len(zrollma.columns),zrollma.columns]
ibs.columns = [['IBS']*len(ibs.columns),ibs.columns]

bars = pd.concat([bars,histreturn,rollma,zrollma,ibs],axis=1)
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


bin, bins = pd.qcut(bars['Close', '^vix'], 10, retbins=True)
bincolumns = [str(round(bins[i],1))+"-"+str(round(bins[i+1],1)) for i in range(0,10)]
bin = bars.groupby(bin).mean()
vixgrp= pd.DataFrame([bin['nreturn1','^gspc'].values, bin['nreturn2','^gspc'].values, bin['nreturn3','^gspc'].values, bin['nreturn4','^gspc'].values, bin['nreturn5','^gspc'].values], columns=bincolumns, index=['D1','D2','D3','D4','D5'])

bin, bins = pd.qcut(bars['IBS', '^gspc'], 10, retbins=True)
bincolumns = [str(round(bins[i],2))+"-"+str(round(bins[i+1],2)) for i in range(0,10)]
bin = bars.groupby(bin).mean()
ibsgrp= pd.DataFrame([bin['nreturn1','^gspc'].values, bin['nreturn2','^gspc'].values, bin['nreturn3','^gspc'].values, bin['nreturn4','^gspc'].values, bin['nreturn5','^gspc'].values], columns=bincolumns, index=['D1','D2','D3','D4','D5'])

bin, bins = pd.qcut(bars['ZRollMA', '^gspc'], 10, retbins=True)
bincolumns = [str(round(bins[i],2))+"-"+str(round(bins[i+1],2)) for i in range(0,10)]
bin = bars.groupby(bin).mean()
zrmagrp= pd.DataFrame([bin['nreturn1','^gspc'].values, bin['nreturn2','^gspc'].values, bin['nreturn3','^gspc'].values, bin['nreturn4','^gspc'].values, bin['nreturn5','^gspc'].values], columns=bincolumns, index=['D1','D2','D3','D4','D5'])


#groups = bars.groupby(pd.cut(bars['IBS', '^gspc'], 10)).mean()

#grp1 = groups['nreturn1','^gspc']



for ticker in symbol:
    d = {ticker:{i: pd.DataFrame() for i in range(1,6)} for ticker in symbol}
    print(ticker)
    for i in range(1,6):
        nretstr = 'nreturn'+str(i)
        nddstr = 'ndrawdown'+str(i)
        stats = d[ticker][i]
        vix = bars['ATRClose', ticker].quantile(0.9)
        belowma = bars[bars['ZRollMA', ticker] < 0][bars['HistReturn', ticker] < 0]
        abovema = bars[bars['ZRollMA', ticker] > 0][bars['HistReturn', ticker] < 0]
        highervol = bars[bars['ZATRClose', ticker] > 0][bars['HistReturn', ticker] < 0]
        lowervol = bars[bars['ZATRClose', ticker] < 0][bars['HistReturn', ticker] < 0]
        highvix = bars[bars['ATRClose', ticker] > vix][bars['HistReturn', ticker] < 0][bars['IBS', ticker]<0.2]
        lowvix = bars[bars['ATRClose', ticker] < vix][bars['HistReturn', ticker] < 0]
        belowmahighvol = belowma[belowma['ZATRClose', ticker] > 2]
        belowmalowvol = belowma[belowma['ZATRClose', ticker] < -1]
        abovemahighvol = abovema[abovema['ZATRClose', ticker] > 2][bars['IBS', ticker]<0.2]
        abovemalowvol = abovema[abovema['ZATRClose', ticker] < -1]
        abovemahighvix = abovema[abovema['ATRClose', ticker]>vix]
        belowmahighvix = belowma[belowma['ATRClose',ticker]>vix]
        belowmahighvolhighvix = belowmahighvol[belowmahighvol['ATRClose',ticker] > vix]
        abovemahighvolhighvix = abovemahighvol[abovemahighvol['ATRClose',ticker] > vix]
        def des(df):
            dfg = pd.Series([df.count(),df[df>0].count()/df.count(),df.mean(), df.std(), df.min(), df.quantile(0.25),df.quantile(0.5),df.quantile(0.75),df.max()], index=['count', 'winct','mean', 'std', 'min', '25%', '50%', '75%', 'max'])
            return dfg
        def desdd(df):
            dfa = pd.Series([df.count(),df[df>df.mean()].count()/df.count(),df.mean(), df.std(), df.min(), df.quantile(0.25),df.quantile(0.5),df.quantile(0.75),df.max()], index=['count', 'winct','mean', 'std', 'min', '25%', '50%', '75%', 'max'])
            return dfa
        def label(stat,name):
            stats[str(name)+str(ticker)+str(i)]=des(stat[nretstr,ticker])
            stats[str(name)+'d'+str(ticker)+str(i)]=desdd(stat[nddstr,ticker])


        stats['baseline'+str(ticker)+str(i)] = des(bars[nretstr,ticker])
        label(bars[bars['HistReturn', ticker] < 0],'basedown1')
        label(bars[bars['IBS', ticker]<0.2],'IBSd')
        label(bars[bars['IBS', ticker]>0.8],'IBSh')
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
        stats['abovema'+str(ticker)+str(i)]=des(abovema[nretstr,ticker])
        stats['belowma'+str(ticker)+str(i)]= des(belowma[nretstr,ticker])


    if ticker == '^gspc': break

summstats = pd.DataFrame([d['^gspc'][1].loc['count'].values,d['^gspc'][1].loc['mean'].values, d['^gspc'][2].loc['mean'].values, d['^gspc'][3].loc['mean'].values,d['^gspc'][4].loc['mean'].values,d['^gspc'][5].loc['mean'].values], index=["Days", "D1", "D2","D3","D4","D5"], columns=d['^gspc'][1].columns)



# list of dataframes
writer = pd.ExcelWriter('outputsheet6.xlsx')
row = 0
for i in range(1,6):
    d['^gspc'][i].to_excel(writer, sheet_name='stats', startrow=row, startcol=0)
    row = row + len(d['^gspc'][i].index) + 2 + 1

summstats.to_excel(writer, sheet_name='stats', startrow=0, startcol=len(d['^gspc'][1].columns)+2)

bars[bars['IBS', ticker]<0.2].to_excel(writer, 'bma')
bars.to_excel(writer, 'overall')
vixgrp.to_excel(writer,'VIX')
ibsgrp.to_excel(writer, 'IBS')
zrmagrp.to_excel(writer, 'zrma')
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

