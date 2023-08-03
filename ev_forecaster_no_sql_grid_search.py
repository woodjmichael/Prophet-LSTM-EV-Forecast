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

# %% [markdown]
# # Options
# 
# - hour-ahead!

# %%
# gpus = tf.config.list_physical_devices('GPU')
# print(gpus)

path = "./Output/Trial_" + time

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
    'n_back': 4*24*3,  # 4*24*7
    'n_timesteps': int(4*24*1.5),  # 96
    'lag': 0,
    'dataset_split': 'data', # 'percentage', data
    'tr_per': 0.90,
    #'training_test_split_data': '30/05/2023 00:15',
    'out_col': ['power'],      #Potenza
    'features': []#['year', 'month', 'day', 'hour', 'minute'],
}

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
             'epochs': 1,
             'validation_split': 0.2,
             'model_path': './Output/',
             'Dropout_rate': 0.2,
             }

# %% [markdown]
# # Data
# 
# ## Import

# %%
filename = 'train_JPL_4_mjw.csv'

dataset = pd.read_csv('dati/'+filename,index_col=0,parse_dates=True)

# dataset = pd.read_csv('dati/'+filename,sep=';')
# dataset['times_utc'] = pd.to_datetime(dataset['times_utc'], format=file_opt['dt_format'])
# dataset.set_index('times_utc', inplace=True)


dataset = dataset.resample('15min').mean()
plot_weekly(dataset.power,title=filename,alpha=0.1)

# %% [markdown]
# ## Clean (optional)
# 
# The second block is run once for each month in the dataset to visually see outliers. 
# 
# Turn on plotly (first cell up top) to easily find the day-of-month to be replaced.

# %%
# year_month = '2020'

# plot_weekly(dataset.loc[year_month].power,alpha=0.3,begin_on_monday=False)
# dataset['weekday'] = dataset.index.weekday*10
# dataset[['weekday','power']].loc[year_month].plot()

# %%
# dataset.loc['2020-12-7':'2020-12-8']  = dataset.loc['2020-11-30':'2020-12-1'].values
# dataset.loc['2021-10-11':'2021-10-15']  = dataset.loc['2021-10-4':'2021-10-8'].values
# dataset.loc['2022-5-23':'2022-5-24']  = dataset.loc['2022-5-16':'2022-5-17'].values
# dataset.loc['2022-9-5':'2022-9-6']  = dataset.loc['2022-8-29':'2022-8-30'].values
# dataset.loc['2023-12-19':'2023-12-22']  = dataset.loc['2023-12-12':'2023-12-15'].values
# dataset.loc['2023-11-20']  = dataset.loc['2023-11-13'].values

# plot_weekly(dataset.power,title='train_JPL_4_mjw.csv Cleaned')
# dataset.to_csv('dati/train_JPL_4_mjw.csv')

# %% [markdown]
# ## Shape

# %%
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

# %%
dump(scaler_X, open(model_opt["model_path"] + "scaler_in.pkl", 'wb'))
dump(scaler_y, open(model_opt["model_path"] + "scaler_out.pkl", 'wb'))

# %% [markdown]
# # Model
# 
# ## Build


data_opt = {
    #'reload_data': False,
    'n_back': 4*24*3,  # 4*24*7
    'n_timesteps': int(4*24*1.5),  # 96
    'lag': 0,
    'dataset_split': 'data', # 'percentage', data
    'tr_per': 0.90,
    #'training_test_split_data': '30/05/2023 00:15',
    'out_col': ['power'],      #Potenza
    'features': []#['year', 'month', 'day', 'hour', 'minute'],
}

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
             'epochs': 1,
             'validation_split': 0.2,
             'model_path': './Output/',
             'Dropout_rate': 0.2,
             }

with open(model_opt['model_path'] + 'Errors.csv', 'w') as f:
    f.write('units1,units2,input_dim,mae\n')
    
for units1 in [12,24,48,96,128,256]:
    for units2 in [12,24,48,96,128,256]:
        units = [units1,units2]
        for input_dim in [24,48,96,144,192,288,672]:
            data_opt['n_back'] = input_dim
            model_opt['LSTM_num_hidden_units'] = units
            model_opt['input_dim'] = (data_opt['n_back'], data_opt['n_features'])
            # %%
            # smltn_chr = 'forecast'

            with open(model_opt['model_path'] + 'Param.txt', 'w') as convert_file:
                convert_file.write(json.dumps(data_opt))
                convert_file.write(json.dumps(model_opt))

            # %% [markdown]
            # ## Train

            # %%
            model_create_load_tune= 'create' # tune , load
            if model_create_load_tune == 'create':
                model, history = create_model_attention(model_opt, train_X, train_y)
                model.save(model_opt['model_path'] + 'model.h5')

            # %% [markdown]
            # # Analyze
            # 
            # ## Load

            # %%
            scaler_X = load(open(model_opt["model_path"] + "scaler_in.pkl", 'rb')) 
            scaler_y = load(open(model_opt["model_path"] + "scaler_out.pkl", 'rb'))

            model = load_model(model_opt["model_path"] + 'model.h5', custom_objects={"attention": attention})

            # %% [markdown]
            # ## Predict

            # %%
            # make a prediction
            yhat_np = model.predict(test_X)
            yhat_np = yhat_np.reshape((yhat_np.shape[0], yhat_np.shape[1]))

            # invert scaling for forecast
            inv_yhat = scaler_y.inverse_transform(yhat_np)

            y_hat = pd.DataFrame(data=inv_yhat, columns=test_y.columns, index=test_y.index)

            # %%

            # %%
            with open(model_opt['model_path'] + 'Errors.csv', 'a') as f:
                mae = (abs(y_hat.values - test_y.values)).mean()
                f.write(f'{units1},{units2},{input_dim},{mae}\n')


