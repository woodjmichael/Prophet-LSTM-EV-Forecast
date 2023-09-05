# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import datetime
import json
from pickle import dump,load
from data_util import prepare_data, plot_weekly
from model_util import create_model_attention, attention
from keras.models import load_model
from pickle import load

plt.interactive(True)
pd.options.plotting.backend='plotly'

current_time = datetime.datetime.now()
time = current_time.strftime("%Y-%m-%d %H:%M:%S%z") #"%Y%m%d_%H%M" impianto

# options
run_dir = "G:/Shared drives/Polimi/Publications/Prophet LSTM EV Forecast/"
sito = 'JPL'         #'impianto4'
file_opt = {
    'file': sito,
    'directory_file': run_dir+"dati/",
    'dt_format': "%d/%m/%Y %H:%M",
    'to_timezone': 'America/Los_Angeles'}   #Europe/Rome
manage_file = {
    'filtra_by_date': True,  # per restituire una porzione di df - insieme a inizio_filtro fine_filtro
    'aggregate_w_we': 'by_day',  # 'grouped' # by_day --> se grouped 0,1 su colonna week_day
    'only_week_weekend': False,  # 'week',  # 'week',  # week weekend False
    'single_day': False,  # 'monday'# False # 'tuesday'
    'remove_nan_ID': False,  # remove session when user ID is not given
    'togli_durata_inferiore': False,}  # se durata sessione inferiore a {minuti} --> rimuovi
data_opt = {
    #'reload_data': False,
    'n_back': 4*24*3,  # 4*24*7
    'n_timesteps': int(4*24*1.5),  # 96
    'lag': 0, # hours
    'dataset_split': 'data', # 'percentage', data
    'tr_per': 0.90,
    #'training_test_split_data': '30/05/2023 00:15',
    'out_col': ['power'],      #Potenza
    'features': []}#['year', 'month', 'day', 'hour', 'minute'],
data_opt['columns'] = data_opt['features'] + data_opt['out_col']
data_opt['n_features'] = len(data_opt['columns'])
model_opt = {'LSTM_num_hidden_units': [24,24],
             #'LSTM_layers': 1,
             'input_dim': (data_opt['n_back'], data_opt['n_features']),
             'dense_out': data_opt['n_timesteps'],
             'neurons_activation': 'relu',
             'metrics': 'mse',
             'optimizer': 'adam',
             'patience': 5,
             'epochs': 50,
             'validation_split': 0.2,
             'model_path': run_dir+'Output/',
             'Dropout_rate': 0.2,}

# data
data_file = 'train_JPL_4_mjw.csv'
dataset = pd.read_csv(file_opt['directory_file']+data_file,index_col=0,parse_dates=True)
dataset = dataset.resample('15min').mean()
dataset['year'] = dataset.index.year
dataset['month'] = dataset.index.month
dataset['day'] = dataset.index.dayofweek
dataset['hour'] = dataset.index.hour
dataset['minute'] = dataset.index.minute
dataset = dataset[data_opt['columns']]

train_X, train_y, test_X, test_y, scaler_X, scaler_y  = prepare_data(dataset, data_opt)
train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1)) #solo ENC-DEC

dump(scaler_X, open(model_opt["model_path"] + "scaler_in.pkl", 'wb'))
dump(scaler_y, open(model_opt["model_path"] + "scaler_out.pkl", 'wb'))

# results file
errors_filepath = model_opt['model_path']+'Errors '+current_time.strftime("%Y-%m-%d")+'.csv'
with open(errors_filepath, 'w') as f:
    f.write('units1,units2,input_dim,mae\n')
    
    
# grid search
for units in [12,24,48,96,128,256]:
    units = [units,units]
    for input_dim in [24,48,96,144,192,288,672]:
        data_opt['n_back'] = input_dim
        model_opt['LSTM_num_hidden_units'] = units
        model_opt['input_dim'] = (data_opt['n_back'], data_opt['n_features'])

        # train
        model_create_load_tune= 'create' # tune , load
        if model_create_load_tune == 'create':
            model, history = create_model_attention(model_opt, train_X, train_y)
            model.save(model_opt['model_path'] + 'model.h5')

        # reload
        scaler_X = load(open(model_opt["model_path"] + "scaler_in.pkl", 'rb')) 
        scaler_y = load(open(model_opt["model_path"] + "scaler_out.pkl", 'rb'))
        model = load_model(model_opt["model_path"] + 'model.h5', custom_objects={"attention": attention})

        # make a prediction
        yhat_np = model.predict(test_X)
        yhat_np = yhat_np.reshape((yhat_np.shape[0], yhat_np.shape[1]))

        # invert scaling for forecast
        inv_yhat = scaler_y.inverse_transform(yhat_np)

        y_hat = pd.DataFrame(data=inv_yhat, columns=test_y.columns, index=test_y.index)

        # save errors
        with open(errors_filepath, 'a') as f:
            mae = (abs(y_hat.values - test_y.values)).mean()
            units1=units[0]
            units2=units[1]
            f.write(f'{units1},{units2},{input_dim},{mae}\n')



