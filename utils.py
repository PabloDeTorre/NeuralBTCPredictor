import pandas as pd
import numpy as np

def add_volatility(data):
    """
    data: input data, pandas DataFrame
    This function calculates the volatility and close_off_high of each given coin in 24 hours, 
    and adds the result as new columns to the DataFrame.
    Return: DataFrame with added columns
    """
    # calculate the daily change
    kwargs = {'Change': lambda x: (x['Close'] - x['Open']) / x['Open'],
              'CloseOffHigh': lambda x: 2*(x['High'] - x['Close']) / (x['High'] - x['Low']) - 1,
              'Volatility': lambda x: (x['High'] - x['Low']) / (x['Open'])}
    data = data.assign(**kwargs)
    return data

def add_EMA(data):
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()

    kwargs = {'MACD': exp1-exp2}
    data = data.assign(**kwargs)
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
    data = data[['Open','High','Low','Close','Volume','Trades','Change','CloseOffHigh','Volatility','MACD']]

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

        if(data[i+(window_len-1),0] > data[i+window_len,0]):
            outputs[i] = 0
        else:
           outputs[i] = 1
        
    return inputs, outputs


def data_normalize(df):
    return (df - df.mean()) / (df.max() - df.min())