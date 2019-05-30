import pandas as pd
import csv
import keras
from keras.models import Model
from keras.layers import Activation, Dense, Dropout
from keras.layers import Input
from keras.layers import CuDNNLSTM, CuDNNGRU
from keras.layers import LeakyReLU

import utils as ut

def createModel(input_shape, output_size, units, activation):
    inp = Input(shape=input_shape)

    lstm1 = CuDNNLSTM(
        units=units,
        return_sequences=True
    )(inp)

    if activation == "leakyRelu":
        activation1 = LeakyReLU(alpha=0.1)(lstm1)
    else:
        activation1 = Activation(activation)(lstm1)
    dropout1 = Dropout(rate=0.25)(activation1)

    lstm2 = CuDNNLSTM(
        units=units
    )(dropout1)

    if activation == "leakyRelu":
        activation2 = LeakyReLU(alpha=0.1)(lstm2)
    else:
        activation2 = Activation(activation)(lstm2)
    dropout2 = Dropout(rate=0.25)(activation2)

    dense = Dense(units=output_size)(dropout2)
    if activation == "leakyRelu":
        activation3 = LeakyReLU(alpha=0.1)(dense)
    else:
        activation3 = Activation(activation)(dense)

    model = Model(inputs=inp, outputs=activation3)
    model.compile(loss='mse', optimizer='adam', metrics=['mae','acc'])
    model.summary()
    return model

# data file path
coinsFile = 'data/CoinAPIBitcoin5min.csv'

# Columns of price data to use
columns = ['Start','End','TimeOpen','TimeClose','Open','High','Low','Close','Volume','Trades']

df = pd.read_csv(coinsFile)
df = ut.add_volatility(df)
df = ut.create_model_data(df)
data = ut.to_array(df)
data = data[200:]

headers = ["window_len", "batch_size", "epochs", "activation", "units", "val_loss", "val_mean_absolute_error", "val_acc", "loss", "mean_absolute_error", "acc"]
result = []

with open('data/resultCSV.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headers)

    batch_size_array = [64, 128, 256]
    epochs = 100
    window_len = 288
    activation_array = ["leakyRelu", "elu", "selu", "softplus", "softsign", "relu", "tanh", "sigmoid", "exponential", "linear"]
    units_array = [64, 128]
    test_size = 0.2

    inputs, outputs = ut.createNonBinaryInputsAndOutputs(data, window_len)

    for batch_size in batch_size_array:
        for activation in activation_array:
            for units in units_array:
                # initialise model architecture
                model = createModel((inputs.shape[1],inputs.shape[2]), output_size=1, units=units, activation=activation)

                # train model on data
                history = model.fit(inputs, outputs, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=False, validation_split=test_size)

                result = [[window_len, batch_size, epochs, activation, units, history.history["val_loss"][epochs - 1], history.history["val_mean_absolute_error"][epochs - 1], history.history["val_acc"][epochs - 1], history.history["loss"][epochs - 1], history.history["mean_absolute_error"][epochs - 1], history.history["acc"][epochs - 1]]]
                writer.writerows(result)

    csvfile.close()
