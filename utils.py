import pandas as pd
import numpy as np
import tulipy as ti

def add_volatility(data):
    kwargs = {'Change': lambda x: (x['Close'] - x['Open']) / x['Open'],
              'CloseOffHigh': lambda x: 2*(x['High'] - x['Close']) / (x['High'] - x['Low']) - 1,
              'Volatility': lambda x: (x['High'] - x['Low']) / (x['Open'])}
    data = data.assign(**kwargs)
    return data

def add_MACD(data):
    
    data['26 ema'] = data['Close'].ewm(span=26).mean()
    data['12 ema'] = data['Close'].ewm(span=12).mean()
    data['MACD'] = (data['12 ema'] - data['26 ema'])
    return data

def add_RSI(data, period):
    delta = data['Close'].diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains
    u = u.drop(u.index[:(period-1)])
    d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses
    d = d.drop(d.index[:(period-1)])
    rs = pd.stats.moments.ewma(u, com=period-1, adjust=False) / \
    pd.stats.moments.ewma(d, com=period-1, adjust=False)

    data['RSI'] = 100 - 100 / (1 + rs)
    return data

def ExpMovingAverage(values, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a =  np.convolve(values, weights, mode='full')[:len(values)]
    a[:window] = a[window]
    return a

def create_model_data(data):
    """
    data: pandas DataFrame
    This function drops unnecessary columns and reverses the order of DataFrame based on decending dates.
    Return: pandas DataFrame
    """
    data = data[['Open','High','Low','Close','Volume','Trades','Change','CloseOffHigh','Volatility']]
    print(data)
    return data


def to_array(data):
    """
    data: DataFrame
    This function will convert list of inputs to a numpy array
    Return: numpy array
    """
    return np.array(data)


def createInputsAndOutputs(data, window_len):
    # -window_len first rows that we can't use
    # -1 last row also we can't use
    inputs = np.zeros((data.shape[0] - window_len, data.shape[1], window_len))
    outputs = np.zeros((data.shape[0] - window_len, 1))

    for i in range(data.shape[0] - window_len - 1):
        inputs[i] = data[i:i+window_len].T

        # Price decreased
        if(data[i+(window_len-1),0] > data[i+window_len,0]):
            outputs[i] = 0
        # Price did not change or increased
        else:
           outputs[i] = 1

    return inputs, outputs

def createNonBinaryInputsAndOutputs(data,window_len):
    # -window_len first rows that we can't use
    # -1 last row also we can't use
    inputs = np.zeros((data.shape[0] - window_len, data.shape[1], window_len))
    outputs = np.zeros((data.shape[0] - window_len, 1))

    for i in range(data.shape[0] - window_len - 1):
        inputs[i] = data[i:i+window_len].T

        # Price down more than 200$
        if(data[i+window_len,0] - data[i+(window_len-1),0] <= -200):
            outputs[i] = 1
        # Price down between 200$ and 50$
        elif(data[i+window_len,0] - data[i+(window_len-1),0] <= -50):
            outputs[i] = 2
        # Price down between 50$ and 0$
        elif(data[i+window_len,0] - data[i+(window_len-1),0] <= 0):
            outputs[i] = 3
        # Price up between 0$ and 50$
        elif(data[i+window_len,0] - data[i+(window_len-1),0] <= 50):
            outputs[i] = 4
        # Price up between 50$ and 200$
        elif(data[i+window_len,0] - data[i+(window_len-1),0] <= 200):
            outputs[i] = 5
        # Price up more than 200$
        else:
            outputs[i] = 6

    return inputs, outputs

def data_normalize(df):
    return (df - df.mean()) / (df.max() - df.min())