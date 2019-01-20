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
        global stt
        stt = 'ST_' + str(self.atr_period) + '_' + str(self.atr_multiplier)
        stx = 'STX_' + str(self.atr_period) + '_' + str(self.atr_multiplier)

        df = self.bars
        df['basic_ub'] = (df['High'] + df['Low']) / 2 + self.atr_multiplier * df[atr]
        df['basic_lb'] = (df['High'] + df['Low']) / 2 - self.atr_multiplier * df[atr]
        df['hplusl'] = (df['High'] + df['Low']) / 2
        df['rollingHL'] = df['hplusl'].rolling(self.pullback_period).mean()
        df['volunit'] = 0.16 * df['TR'].rolling(self.pullback_period).mean()
        df['valueprice'] = (df['Low'] - df['rollingHL']) / df['volunit']
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
        df[stt] = 0.00
        for i in range(self.atr_period, len(df)):
            df[stt].iat[i] = df['final_ub'].iat[i] if df[stt].iat[i - 1] == df['final_ub'].iat[i - 1] and \
                                                      df['Close'].iat[
                                                          i] <= df['final_ub'].iat[i] else \
                df['final_lb'].iat[i] if df[stt].iat[i - 1] == df['final_ub'].iat[i - 1] and df['Close'].iat[i] > \
                                         df['final_ub'].iat[i] else \
                    df['final_lb'].iat[i] if df[stt].iat[i - 1] == df['final_lb'].iat[i - 1] and df['Close'].iat[i] >= \
                                             df['final_lb'].iat[i] else \
                        df['final_ub'].iat[i] if df[stt].iat[i - 1] == df['final_lb'].iat[i - 1] and df['Close'].iat[
                            i] < \
                                                 df['final_lb'].iat[i] else 0.00

            # Mark the trend direction up/down
        df[stx] = np.where((df[stt] > 0.00), np.where((df['Close'] < df[stt]), -1, 1), np.NaN)
        df['stxdiff'] = df[stx].diff()

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
            if df[stx][i] - df[stx][i - 1] != 0:
                highcount[i] = 0
                uptcounter[i] = 0
                lowcount[i] = 0
                downtcounter[i] = 0
            elif (df['High'][i] > df['High'][i - 1]) and (df[stx][i] == 1):
                highcount[i] = highcount[i - 1] + 1
            elif (df['Low'][i] < df['Low'][i - 1]) and (df[stx][i] == -1):
                lowcount[i] = lowcount[i - 1] + 1
            elif (df['Low'][i] < df['Low'][i - 1]) and (df[stx][i] == 1):
                uptcounter[i] = uptcounter[i - 1] + 1
            else:
                downtcounter[i] = downtcounter[i - 1] + 1

        df['highcount'] = highcount
        df['uptcounter'] = uptcounter
        df['lowcount'] = lowcount
        df['downtcounter'] = downtcounter
        return df

    def entry(self, ctperiod=2, rr=3):
        df = self.bars
        longentrysig = pd.Series(index=df.index).fillna(0)
        longexitsig = pd.Series(index=df.index).fillna(0)
        entrypx = pd.Series(index=df.index).fillna(0)
        exitpx = pd.Series(index=df.index).fillna(0)
        stoppx = pd.Series(index=df.index).fillna(0)
        targetpx = pd.Series(index=df.index).fillna(0)
        stoptrig = pd.Series(index=df.index)
        targettrig = pd.Series(index=df.index)
        tradestatus = pd.Series(index=df.index).fillna(0)

        for i in range(1, len(df.index)):
            # criteria for triggering an entry signal: 1. stx must be positive, 2. no trades currently on, 3. more than 3 prior countertrend moves, 4. current bar Higher high
            if df[stx][i] == 1 and tradestatus[i - 1] == 0 and any(df['uptcounter'][i-1:i] >= ctperiod) and \
                    df['High'][i] > df['High'][i - 1]:
                longentrysig[i] = 1
            else:
                longentrysig[i] = 0
            if longentrysig[i] == 1:
                # upon entry logic, set entry, stop, target prices
                entrypx[i] = max(df['High'][i - 1], df['Low'][i])
                stoppx[i] = max(df['Low'][i - 1], df[stx][i - 1])
                targetpx[i] = entrypx[i] + (entrypx[i] - stoppx[i]) * rr
                tradestatus[i] = 1
                ##stoplogic on the current bar
                if stoppx[i] < df['Low'][i]:
                    stoptrig.iloc[i] = "n"
                else:
                    stoptrig.iloc[i] = "y"
                    exitpx[i] = stoppx[i]
                if targetpx[i] > df['High'][i]:
                    targettrig.iloc[i] = "n"
                else:
                    targettrig.iloc[i] = "y"
                    exitpx[i] = targetpx[i]
            elif tradestatus[i - 1] == 1:
                entrypx[i] = entrypx[i - 1]
                stoppx[i] = stoppx[i - 1]
                targetpx[i] = targetpx[i - 1]
                # stop logic on subsequent bars
                if stoppx[i] < df['Low'][i]:
                    stoptrig.iloc[i] = "n"
                else:
                    stoptrig.iloc[i] = "y"
                    exitpx[i] = stoppx[i]
                if targetpx[i] > df['High'][i]:
                    targettrig.iloc[i] = "n"
                else:
                    targettrig.iloc[i] = "y"
                    exitpx[i] = targetpx[i]
            else:
                entrypx[i] = 0
                stoppx[i] = 0
                targetpx[i] = 0

            ##tradestatus
            if (targettrig.iloc[i] == "y") or (stoptrig.iloc[i] == "y"):
                longexitsig[i] = -1
            if (longentrysig[i] == 1) and (targettrig.iloc[i] == "n") and (stoptrig.iloc[i] == "n"):
                tradestatus[i] = 1
            elif (targettrig.iloc[i] == "n") and (stoptrig.iloc[i] == "n"):
                tradestatus[i] = tradestatus[i - 1]
            else:
                tradestatus[i] = 0

        df['longentrysig'] = longentrysig
        df['longexitsig'] = longexitsig
        df['entrypx'] = entrypx
        df['exitpx'] = exitpx
        df['stoppx'] = stoppx
        df['targetpx'] = targetpx
        df['stoptrig'] = stoptrig
        df['targettrig'] = targettrig
        df['tradestatus'] = tradestatus
        return df


class PortfolioGenerate(Portfolio):
    """Encapsulates the notion of a portfolio of positions based
    on a set of signals as provided by a Strategy.

    Requires:
    symbol - A stock symbol which forms the basis of the portfolio.
    bars - A DataFrame of bars for a symbol set.
    signals - A pandas DataFrame of signals (1, 0, -1) for each symbol.
    initial_capital - The amount in cash at the start of the portfolio."""

    def __init__(self, symbol, bars, signals, initial_capital=100000.0, risk=0.02):
        self.symbol = symbol
        self.bars = bars
        self.signals = signals
        self.initial_capital = float(initial_capital)
        self.risk = risk

    def backtest_portfolio(self):
        global portfolio
        portfolio = pd.DataFrame(index=self.bars.index).fillna(0)
        portfolio['longentrysig'] = self.bars['longentrysig']
        portfolio['longexitsig'] = self.bars['longexitsig']
        portfolio['tradestatus'] = self.bars['tradestatus']
        portfolio['entrypx'] = self.bars['entrypx']
        portfolio['exitpx'] = self.bars['exitpx']
        portfolio['ClosePx'] = self.bars['Close']
        entrytrade = pd.Series(index=self.bars.index).fillna(0)
        closetradeholding = pd.Series(index=self.bars.index).fillna(0)
        exittrade = pd.Series(index=self.bars.index).fillna(0)

        for i in range(1, len(portfolio.index)):
            if portfolio['longentrysig'][i] == 1:
                suggestedpositionsize = self.initial_capital * self.risk / (
                            self.bars['entrypx'][i] - self.bars['stoppx'][i])
                closetradeholding[i] = suggestedpositionsize * (
                            portfolio['longentrysig'][i] + portfolio['longexitsig'][i])
                entrytrade[i] = self.bars['entrypx'][i] * suggestedpositionsize
                exittrade[i] = portfolio['exitpx'][i] * suggestedpositionsize
            elif portfolio['longexitsig'][i] == -1:
                exittrade[i] = portfolio['exitpx'][i] * suggestedpositionsize
                closetradeholding[i] = 0
            else:
                closetradeholding[i] = closetradeholding[i - 1]

        portfolio['trades'] = -entrytrade + exittrade
        portfolio['positiononclose'] = closetradeholding
        portfolio['holdings'] = portfolio['positiononclose'] * portfolio['ClosePx']
        portfolio['cash'] = self.initial_capital + (portfolio['trades']).cumsum()
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change()
        return portfolio

class Analytics():

    def summaryanalytics(self):
        ##Analytics
        entry = pd.Series(portfolio['entrypx'].loc[portfolio.longentrysig == 1.0].values)
        stop = pd.Series(signals['stoppx'].loc[portfolio.longentrysig == 1.0].values)
        exit = pd.Series(portfolio['exitpx'].loc[portfolio.longexitsig == -1.0].values)
        entrydate = pd.Series(portfolio.loc[portfolio.longentrysig == 1.0].index)
        exitdate = pd.Series(portfolio.loc[portfolio.longexitsig == -1.0].index)
        favexcursion = [signals['High'].loc[x:y].max() for x, y in zip(entrydate, exitdate)]
        adverseexcursion = [signals['Low'].loc[x:y].min() for x, y in zip(entrydate, exitdate)]
        tradeduration = [len(signals.loc[x:y].index) for x, y in zip(entrydate, exitdate)]

        ##SummaryTable
        SummaryTable = pd.concat(
            [pd.Series(entry, name="EntryPx"), pd.Series(stop, name="StopPx"), pd.Series(exit, name="ExitPx"),
             pd.Series(favexcursion, name="MFE"), pd.Series(adverseexcursion, name="MAE"),
             pd.Series(entrydate, name="EntryDate"), pd.Series(exitdate, name="exitdate"),
             pd.Series(tradeduration, name="TradeDur")], axis=1)

        SummaryTable['Risk%'] = 100 * ((SummaryTable['StopPx'] / SummaryTable['EntryPx']) - 1)
        SummaryTable['Returns%'] = 100 * ((SummaryTable['ExitPx'] / SummaryTable['EntryPx']) - 1)
        SummaryTable['MFE RR'] = -100 * ((SummaryTable['MFE'] / SummaryTable['EntryPx']) - 1) / SummaryTable['Risk%']
        SummaryTable['MAE RR'] = -100 * ((SummaryTable['MAE'] / SummaryTable['EntryPx']) - 1) / SummaryTable['Risk%']
        SummaryTable.to_csv('summstats.csv')

        winners = len(SummaryTable[SummaryTable['Returns%'] > 0])
        winavgmfe = SummaryTable['MFE RR'].loc[SummaryTable['Returns%'] > 0].median()
        winavgmae = SummaryTable['MAE RR'].loc[SummaryTable['Returns%'] > 0].median()
        loseavgmfe = SummaryTable['MFE RR'].loc[SummaryTable['Returns%'] < 0].median()
        loseavgmae = SummaryTable['MAE RR'].loc[SummaryTable['Returns%'] < 0].median()
        windur = SummaryTable['TradeDur'].loc[SummaryTable['Returns%'] > 0].median()
        losedur = SummaryTable['TradeDur'].loc[SummaryTable['Returns%'] < 0].median()
        startdate = signals.index[0]
        enddate = signals.index[-1]
        cagr = 100 * ((portfolio['total'].iloc[-1] / 100000) ** (1 / ((enddate - startdate).days / 365)) - 1)
        bhcagr = 100 * ((portfolio['ClosePx'].iloc[-1] / portfolio['ClosePx'].iloc[0]) ** (1 / ((enddate - startdate).days / 365)) - 1)

        print("On {}, Acc Equity compounded at {}% CAGR vs BuyHold at {}% CAGR".format(symbol, round(cagr, 2),round(bhcagr,2)))
        print("Winrate is {}% with {} winners".format(round(100 * winners / len(SummaryTable), 1), winners))
        print("Winner MFE is {} R, MAE is {} R".format(round(winavgmfe, 2), round(winavgmae, 2)))
        print("Loser MFE is {} R, MAE is {} R".format(round(loseavgmfe, 2), round(loseavgmae, 2)))
        print("Winner duration {} days, Loser duration {} days".format(windur, losedur))

        return SummaryTable

if __name__ == "__main__":
    # Obtain daily bars of AAPL from Yahoo Finance for the period
    symbol = 'TSLA'
    bars = web.DataReader(symbol, 'yahoo', datetime.datetime(2010, 1, 1), datetime.datetime(2019, 1, 1))
    st = STStrategy(symbol, bars, atr_period=10, atr_multiplier=3, pullback_period=5)
    signals = st.ATR()
    signals = st.SuperTrend()
    signals = st.HLCounter()
    signals = st.entry(ctperiod=1, rr=3)
    portfolio = PortfolioGenerate(symbol, bars, signals, initial_capital=100000.0)
    returns = portfolio.backtest_portfolio()
    Analytics().summaryanalytics()
    returns.to_csv('test6.csv')


    # Plot two charts to assess trades and equity curve
    fig = plt.figure()
    fig.patch.set_facecolor('white')  # Set the outer colour to white
    ax1 = fig.add_subplot(211, ylabel='Price in $')

    # Plot the AAPL closing price overlaid with the indicator
    bars['Close'].plot(ax=ax1, color='r', lw=2.)
    signals[[stt]].plot(ax=ax1, lw=2.)

    # Plot the "buy" trades against AAPL
    ax1.plot(portfolio.loc[portfolio.longentrysig == 1.0].index,
             portfolio['entrypx'].loc[portfolio.longentrysig == 1.0],
             '^', markersize=5, color='m')

    # Plot the "exit" trades against AAPL
    ax1.plot(portfolio.loc[portfolio.longexitsig == -1.0].index,
             portfolio['exitpx'].loc[portfolio.longexitsig == -1.0],
             '^', markersize=5, color='k')

    # Plot the equity curve in dollars
    ax2 = fig.add_subplot(212, ylabel='Portfolio value in $')
    returns['total'].plot(ax=ax2, lw=2.)

    # Plot the figure
    plt.show()




