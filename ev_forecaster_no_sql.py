# %% build
#
#

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import datetime
import json
from data_util import prepare_data
from model_util import create_model_attention

 

# gpus = tf.config.list_physical_devices('GPU')
# print(gpus)
plt.interactive(True)
 

current_time = datetime.datetime.now()
time = current_time.strftime("%Y-%m-%d %H:%M:%S%z") #"%Y%m%d_%H%M" impianto
path = "./Output/Trial_" + time
#os.mkdir(path)

 

print('definizione parametri')
# load dataset
sito = 'JPL'         #'impianto4'

 

file_opt = {
    'file': sito,
    'directory_file': "./dati/",
    'dt_format': "%d/%m/%Y %H:%M",
    'to_timezone': 'America/Los_Angeles'}   #Europe/Rome

 

manage_file = {
    'filtra_by_date': True,  # per restituire una porzione di df - insieme a inizio_filtro fine_filtro
    'aggregate_w_we': 'by_day',  # 'grouped' # by_day --> se grouped 0,1 su colonna week_day
    'only_week_weekend': False,  # 'week',  # 'week',  # week weekend False
    'single_day': False,  # 'monday'# False # 'tuesday'
    'remove_nan_ID': False,  # remove session when user ID is not given
    'togli_durata_inferiore': False,  # se durata sessione inferiore a {minuti} --> rimuovi
}
# scaled = values
# specify the number of lag hours
data_opt = {
    #'reload_data': False,
    'n_back': 96*3,  # 4*24*7
    'n_timesteps': 4,  # 4*4
    'lag': 0,
    'dataset_split': 'data', # 'percentage', data
    'tr_per': 0.90,
    #'training_test_split_data': '30/05/2023 00:15',
    'out_col': ['power'],      #Potenza
    'features': []#['year', 'month', 'day', 'hour', 'minute'],
}

 

# smltn_chr = 'forecast'

 

data_opt['columns'] = data_opt['features'] + data_opt['out_col']
data_opt['n_features'] = len(data_opt['columns'])

 

model_opt = {'LSTM_num_hidden_units': [24],
             #'LSTM_layers': 1,
             'input_dim': (data_opt['n_back'], data_opt['n_features']),
             'dense_out': data_opt['n_timesteps'],
             'neurons_activation': 'relu',
             'metrics': 'mse',
             'optimizer': 'adam',
             'patience': 5,
             'epochs': 30,
             'validation_split': 0.2,
             'model_path': './Output/',
             'Dropout_rate': 0.2,
             }

with open(model_opt['model_path'] + 'Param.txt', 'w') as convert_file:
    convert_file.write(json.dumps(data_opt))
    convert_file.write(json.dumps(model_opt))

#%%
dataset = pd.read_csv('dati/train_JPL_2_pressure_washed.csv',sep=';')


#data_opt['training_test_split_data'] = pd.to_datetime(data_opt['training_test_split_data'], format=file_opt['dt_format'])

 
dataset['year'] = dataset.index.year
dataset['month'] = dataset.index.month
dataset['day'] = dataset.index.dayofweek
dataset['hour'] = dataset.index.hour
dataset['minute'] = dataset.index.minute

 

dataset = dataset[data_opt['columns']]

 

# print('data loaded')
train_X, train_y, test_X, test_y, scaler_X, scaler_y  = prepare_data(dataset, data_opt)
train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1)) #solo ENC-DEC
# a = np.load('dati_veicoli/train_X.npy')

# %% train
#
#

model_create_load_tune= 'create' # tune , load
if model_create_load_tune == 'create':
    model, history = create_model_attention(model_opt, train_X, train_y)
    model.save(model_opt['model_path'] + 'my_model/microgrid_model.h5')
    # plot history


    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig(model_opt['model_path'] + 'history.png')

 

 # %% predict
 #
 #

# make a prediction
yhat_np = model.predict(test_X)
yhat_np = yhat_np.reshape((yhat_np.shape[0], yhat_np.shape[1]))
# invert scaling for forecast
inv_yhat = scaler_y.inverse_transform(yhat_np)

y_hat = pd.DataFrame(data=inv_yhat, columns=test_y.columns, index=test_y.index)

y_hat.to_csv(f"{model_opt['model_path']}forecast.csv")
test_y.to_csv(f"{model_opt['model_path']}measures.csv")

# %% plot
#
#

idx = pd.date_range(y_hat.index[0],periods=168,freq='1h')
y_hat_flat = y_hat[['out1(t+0)','out1(t+1)','out1(t+2)','out1(t+3)']].loc[idx].values.flatten()

test_y_flat = test_y[['out1(t+0)','out1(t+1)','out1(t+2)','out1(t+3)']].loc[idx].values.flatten()

pd.DataFrame({'y_hat':y_hat_flat,'test_y':test_y_flat},
            index=pd.date_range(idx[0],periods=len(y_hat_flat),freq='15min')).plot()

