import json
import numpy as np
import urllib.request as urllib2
import pandas as pd

# connect to Poloniex's API
url2016 = 'https://poloniex.com/public?command=returnChartData&currencyPair=USDT_BTC&start=1451606400&end=1483227000&period=1800'
url2017 = 'https://poloniex.com/public?command=returnChartData&currencyPair=USDT_BTC&start=1483228800&end=1514763000&period=1800'
url2018toNow = 'https://poloniex.com/public?command=returnChartData&currencyPair=USDT_BTC&start=1514764800&end=999999999999&period=1800'

# parse json returned from Poloniex to Pandas DF
openUrl = urllib2.urlopen(url2018toNow)
r = openUrl.read()
openUrl.close()
d = json.loads(r.decode())
df = pd.DataFrame(d)

original_columns=[u'close', u'date', u'high', u'low', u'open', u'volume', u'weightedAverage']
new_columns = ['Close','Timestamp','High','Low','Open','Volume','WeightedAverage']
df = df.loc[:,original_columns]
df.columns = new_columns
with open('data/bitcoin.csv', 'a') as f:
    df.to_csv(f, index=False, header=False) #Delete headers=False when creating the file for first time