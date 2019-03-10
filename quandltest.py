import quandl


quandl.ApiConfig.api_key = "zFuFxv3joR9sQ7uYLYoZ"

mydata = quandl.get(["CHRIS/CME_FF1.6","CHRIS/CME_FF2.6", "CHRIS/CME_FF3.6","FRED/DFF"], start_date="1995-12-31", end_date="2019-03-05", collapse="weekly")
mydata = mydata.dropna(subset=['CHRIS/CME_FF1 - Settle'])  # to drop if all values in the row are nan


ratehike = 0.25
mydata['1mthprob']=(100-mydata['CHRIS/CME_FF1 - Settle']-mydata['FRED/DFF - Value'])/ratehike
mydata['2mthprob']=(100-mydata['CHRIS/CME_FF2 - Settle']-mydata['FRED/DFF - Value'])/ratehike
mydata['3mthprob']=(100-mydata['CHRIS/CME_FF3 - Settle']-mydata['FRED/DFF - Value'])/ratehike


mydata.to_csv('qdata.csv')
