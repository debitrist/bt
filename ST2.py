import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from backtest import Strategy, Portfolio

class STStrategy(Strategy):
    """
    Requires:
    symbol - A stock symbol on which to form a strategy on.
    bars - A DataFrame of bars for the above symbol.
    short_window - Lookback period for short moving average.
    long_window - Lookback period for long moving average."""

    def __init__(self, symbol, bars, atr_period=10, atr_multiplier=3, pullback_period=5):
        self.symbol = symbol
        self.bars = bars

        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.pullback_period = pullback_period

    def ATR(self):
        global atr
        atr = 'ATR_' + str(self.atr_period)
        df = self.bars
        df['h-l'] = df['High'] - df['Low']
        df['h-yc'] = abs(df['High'] - df['Close'].shift())
        df['l-yc'] = abs(df['Low'] - df['Close'].shift())
        df['TR'] = df[['h-l', 'h-yc', 'l-yc']].max(axis=1)
        df[atr] = df['TR'].rolling(self.atr_period).mean()
        df[atr] = (df[atr].shift() * (self.atr_period - 1) + df['TR']) / self.atr_period
        df.drop(['h-l', 'h-yc', 'l-yc'], inplace=True, axis=1)
        return df

    def SuperTrend(self):
        global stx
        st = 'ST_' + str(self.atr_period) + '_' + str(self.atr_multiplier)
        stx = 'STX_' + str(self.atr_period) + '_' + str(self.atr_multiplier)

        df=self.bars
        df['basic_ub'] = (df['High'] + df['Low']) / 2 + self.atr_multiplier * df[atr]
        df['basic_lb'] = (df['High'] + df['Low']) / 2 - self.atr_multiplier * df[atr]
        df['hplusl'] = (df['High'] + df['Low']) / 2
        df['rollingHL'] = df['hplusl'].rolling(self.pullback_period).mean()
        df['volunit'] = 0.16*df['TR'].rolling(self.pullback_period).mean()
        df['valueprice'] = (df['Low']-df['rollingHL'])/df['volunit']
        # Compute final upper and lower bands
        df['final_ub'] = 0.00
        df['final_lb'] = 0.00
        for i in range(self.atr_period, len(df)):
            df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or \
                                                             df['Close'].iat[i - 1] > df['final_ub'].iat[i - 1] else \
                df['final_ub'].iat[i - 1]
            df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or \
                                                             df['Close'].iat[i - 1] < df['final_lb'].iat[i - 1] else \
                df['final_lb'].iat[i - 1]

        # Set the Supertrend value
        df[st] = 0.00
        for i in range(self.atr_period, len(df)):
            df[st].iat[i] = df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['Close'].iat[
                i] <= df['final_ub'].iat[i] else \
                df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['Close'].iat[i] > \
                                         df['final_ub'].iat[i] else \
                    df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['Close'].iat[i] >= \
                                             df['final_lb'].iat[i] else \
                        df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['Close'].iat[i] < \
                                                 df['final_lb'].iat[i] else 0.00

            # Mark the trend direction up/down
        df[stx] = np.where((df[st] > 0.00), np.where((df['Close'] < df[st]), -1, 1), np.NaN)
        df['stxdiff'] =df[stx].diff()

        # Remove basic and final bands from the columns
        df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)
        df.fillna(0.00, inplace=True)
        return df

    def HLCounter(self):
        df = self.bars
        highcount = pd.Series(index=df.index).fillna(0.0)
        uptcounter = pd.Series(index=df.index).fillna(0.0)
        lowcount = pd.Series(index=df.index).fillna(0.0)
        downtcounter = pd.Series(index=df.index).fillna(0.0)

        for i in range(1, len(df.index)):
            if df[stx][i]-df[stx][i-1] != 0:
                highcount[i]=0
                uptcounter[i]=0
                lowcount[i]=0
                downtcounter[i]=0
            elif (df['High'][i] > df['High'][i-1]) and (df[stx][i] == 1):
                highcount[i]=highcount[i-1]+1
            elif (df['Low'][i] < df['Low'][i-1]) and (df[stx][i] == -1):
                lowcount[i]=lowcount[i-1]+1
            elif (df['Low'][i] < df['Low'][i-1]) and (df[stx][i] == 1):
                uptcounter[i]=uptcounter[i-1]+1
            else: downtcounter[i]=downtcounter[i-1]+1

        global df1
        df1=pd.DataFrame(pd.Series(highcount))
        df['highcount'] = highcount
        df['lowcount'] = lowcount
        df['uptcounter'] = uptcounter
        df['downtcounter'] = downtcounter
        return df


if __name__ == "__main__":
    # Obtain daily bars of AAPL from Yahoo Finance for the period
    symbol = 'AAPL'
    bars = web.DataReader(symbol, 'yahoo', datetime.datetime(2010, 1, 1), datetime.datetime(2019, 1, 1))

    st = STStrategy(symbol, bars, atr_period=10, atr_multiplier=3, pullback_period=5)
    signals = st.ATR()
    signals = st.SuperTrend()
    signals = st.HLCounter()
    print(df1)
    print(signals)
    signals.to_csv('test3.csv')
