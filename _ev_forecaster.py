import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pickle import load
from tensorflow import keras

from PROPHET_EV.util import util
from PROPHET_EV.util.data_util import prepare_data_test
from PROPHET_EV.util.model_util import attention

from PROPHET_DB import mysql

# import daiquiri
# import sklearn.preprocessing

# from PROPHET_DB.setup_logger import setup_logger

# Logger definition
# LOGGER = daiquiri.getLogger(__name__)


def main(argv=None):
    # read configuration file
    args = util.parse_arguments(argv)

    config = util.read_config(args.config)
    data_opt = {
        'n_back': config.getint("data_opt", "n_back"),  # 4*24*7
        'n_timesteps': config.getint("data_opt", "n_timesteps"),  # 4*4
        'lag': config.getint("data_opt", "lag"),
        'tr_per': config.getfloat("data_opt", "tr_per"),
        'out_col': config.get("data_opt", "out_col").split(','),
        'features': config.get("data_opt", "features").split(','),
        'freq': config.getint("data_opt", "freq"),
        'tr_days_step': config.getint("data_opt", "tr_days_step"),
    }
    data_opt['columns'] = data_opt['features'] + data_opt['out_col']
    data_opt['n_features'] = len(data_opt['columns'])

    model_opt = {'Dense_input_dim': config.getint("model_opt", "Dense_input_dim"),
                 'LSTM_num_hidden_units': list(map(int, config.get("model_opt", "LSTM_num_hidden_units").split(','))),
                 'LSTM_layers': config.getint("model_opt", "LSTM_layers"),
                 'metrics': config.get("model_opt", "metrics"), 'optimizer': config.get("model_opt", "optimizer"),
                 'patience': config.getint("model_opt", "patience"), 'epochs': config.getint("model_opt", "epochs"),
                 'validation_split': config.getfloat("model_opt", "validation_split"),
                 'model_path': config.get("model_opt", "model_path"),
                 'Dropout_rate': config.getfloat("model_opt", "Dropout_rate"),
                 'input_dim': (data_opt['n_back'], data_opt['n_features']), 'dense_out': data_opt['n_timesteps']}

    # logger_config = {'mailhost': (config["logger"]["mailhost"], config.getint("logger", "port")),
    #                  'fromaddr': config["logger"]["fromaddr"],
    #                  'toaddr': config["logger"]["toaddrs"],
    #                  'subject': config["logger"]["subject"],
    #                  'credentials': (config["logger"]["username"], config["logger"]["password"]),
    #                  'mail_handler': config.getboolean("logger", "mail_handler"),
    #                  'backup_count': config.getint("logger", "backup_count")}

    #
    talktoSQL = mysql.MySQLConnector(database=config["mysql"]["database"],
                                     host=config["mysql"]["host"],
                                     port=config["mysql"]["port"],
                                     user=config["mysql"]["user"],
                                     password=config["mysql"]["password"])
    days_back = 10
    # ev_query = util.create_query(talktoSQL, config["sql_table"]["ev_power_table"], config["sql_table"]["time_column"])
    ev_query_test = util.create_query_test(talktoSQL, config["sql_table"]["ev_power_table"], config["sql_table"]["time_column"],days_back)
    df = talktoSQL.read_query(ev_query_test, {config["sql_table"]["time_column"]})
    df[config["sql_table"]["time_column"]] = pd.to_datetime(df[config["sql_table"]["time_column"]],
                                                            format='%Y-%m-%d %H:%M:%S%z')#.dt.tz_convert('America/Los_Angeles')

    df.rename(columns={config["sql_table"]["ev_power_field"]: 'power'}, inplace=True)
    df.set_index(config["sql_table"]["time_column"], inplace=True)

    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.dayofweek
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df = df[data_opt['columns']]

    ## definizione tempi inizio
    now = datetime.utcnow()
    inizio_ts = pd.to_datetime(now).tz_localize('UTC')-timedelta(hours=(data_opt['n_back']/4+0.25))  # 7 per aggiustare ora LA, modificare
    fine_ts = pd.to_datetime(now).tz_localize('UTC')

    mask_ts = np.logical_and(df.index > inizio_ts, df.index < fine_ts)
    test = df.loc[mask_ts]
    #### load modello e scaler

    scaler_X = load(open(model_opt["model_path"] + "scaler_in.pkl", 'rb'))
    scaler_y = load(open(model_opt["model_path"] + "scaler_out.pkl", 'rb'))

    test_X, test_y = prepare_data_test(test, scaler_X, data_opt)
    # test_y = test_y.reshape((test_y.shape[0], test_y.shape[1], 1))
    model = keras.models.load_model(model_opt["model_path"] + 'model.h5', custom_objects={"attention": attention})
    y_hat_sc = model.predict(test_X)
    y_hat_sc = y_hat_sc.reshape((test_X.shape[0], test_X.shape[1]))
    y_hat = scaler_y.inverse_transform(y_hat_sc)

    mask_neg_nan = np.logical_or(y_hat < 0, np.isnan(y_hat))
    y_hat[mask_neg_nan] = 0
    y_hat[y_hat > 25] = 25

    # 'id, timestamp_utc, predicted_activepower_ev_1, timestamp_forecast_update'
    forecast_dict = {'timestamp_utc': pd.date_range(start=test_y.index[-1], periods=data_opt['n_timesteps'], freq='15T'),
                     'predicted_activepower_ev_1': y_hat[-1, :],
                     'timestamp_forecast_update': datetime.utcnow()}

    forecast_df = pd.DataFrame.from_dict(forecast_dict)

    talktoSQL.write(forecast_df, config["sql_table"]["ev_forecast_table"])


if __name__ == "__main__":
    # data_opt = {
    #     'reload_data': False,
    #     'n_back': 672,  # 4*24*7
    #     'n_timesteps': 1,  # 4*4
    #     'lag': 96,
    #     'tr_per': 0.8,
    #     'out_col': ['power'],
    #    'features': ['year', 'month', 'day', 'hour', 'day'],
    #   'freq': 15,
    #   'tr_days_step': 7,
    # }
    # data_opt['columns'] = data_opt['features'] + data_opt['out_col']
    # data_opt['n_features'] = len(data_opt['columns'])

    # model_opt = {'Dense_input_dim': 256,
    #              'LSTM_num_hidden_units': [512, 256, 512, 512, 256],
    #             'LSTM_layers': 4,
    #            'input_dim': (data_opt['n_back'], data_opt['n_features']),
    #            'dense_out': data_opt['n_timesteps'],
    #            'neurons_activation': 'relu',
    #            'metrics': 'mse',
    #            'optimizer': 'adam',
    #            'patience': 15,
    #            'epochs': 100,
    #            'validation_split': 0.2,
    #            'model_path': 'model/',
    #            'Dropout_rate': 0.2,
    #            }

    # data_opt['offset_hours'] = (data_opt['n_back']) / 4  # (data_opt['n_back'] + data_opt['lag'])/4

    main()