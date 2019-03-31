import pandas as pd

import keras
from keras.models import Model
from keras.layers import Activation, Dense
from keras.layers import Input
from keras.layers import CuDNNLSTM, CuDNNGRU
from keras.layers import LeakyReLU

import utils as ut

def build_model(input_shape, output_size):
    inp = Input(shape=input_shape)

    lstm = CuDNNLSTM(
        units=32
    )(inp)

    activation1 = LeakyReLU(alpha=0.1)(lstm)
    """
    gru = CuDNNGRU(
        units=32
    )(activation1)
    """

    # lstm2 = CuDNNLSTM(
    #     units=32
    # )(lstm1)

    dense = Dense(units=output_size)(activation1)
    # activation = Activation('relu')(dense)
    activation2 = LeakyReLU(alpha=0.1)(dense)
    model = Model(inputs=inp, outputs=activation2)
    model.compile(loss='mse', optimizer='adam', metrics=['mae','acc'])
    model.summary()
    return model

# data file path
#coinsFile = 'data/bitcoin.csv'
coinsFile = 'data/CoinAPIBitcoin5min.csv'

# Columns of price data to use
columns = ['Start','End','TimeOpen','TimeClose','Open','High','Low','Close','Volume','Trades']

df = pd.read_csv(coinsFile).loc[:, columns]
df = ut.add_volatility(df)
df = ut.add_EMA(df)
df = ut.create_model_data(df)
data = ut.to_array(df)
data = data[200:]

batch_size = 128
epochs = 100
window_len = 288
test_size = 0.2

inputs, outputs = ut.createInputsAndOutputs(data, window_len)

# initialise model architecture
model = build_model((inputs.shape[1],inputs.shape[2]), output_size=1)

# train model on data
history = model.fit(inputs, outputs, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=False, validation_split=test_size)