import requests
import sys

import json
from pandas.io.json import json_normalize
import pandas as pd

URL = 'https://rest.coinapi.io/v1/ohlcv/BTC/USD/history'
#HEADERS = {'X-CoinAPI-Key' : '08FE08AF-652B-4F87-B7BA-4A2FD8750FFA'}
#HEADERS = {'X-CoinAPI-Key' : 'E30AAFEE-A480-430C-A9A5-3B60465B3767'}
#HEADERS = {'X-CoinAPI-Key' : '0187EA37-9345-45DE-8073-5CA779C79326'}
HEADERS = {'X-CoinAPI-Key' : '485927F2-7667-4D25-BDC2-7AEF495558FE'}
PARAMS = { 'period_id':'5MIN', 'time_start':'2018-06-25T00:10:00', 'limit':77760}

if len(sys.argv) == 2:
    outputFile = sys.argv[1]
    r = requests.get(url = URL, headers = HEADERS, params = PARAMS)
    data = r.json()

    # converting json dataset from dictionary to dataframe
    df = pd.DataFrame.from_dict(data, orient='columns')

    original_columns=[u'time_period_start',u'time_period_end',u'time_open',u'time_close',u'price_open',u'price_high',u'price_low',u'price_close',u'volume_traded',u'trades_count']
    new_columns = ['Start','End','TimeOpen','TimeClose','Open','High','Low','Close','Volume','Trades']

    df = df.loc[:,original_columns]
    df.columns = new_columns
    #df.to_csv('data/CoinAPIBitcoin5min.csv', index=False)
    df.to_csv('data/'+ outputFile +'.csv', index=False)
    print("Terminado")
else:
    print("ERROR")

# Habría que hacer que el 'time_start' se fuera actualizando con el último 'timestamp' de "data.txt"