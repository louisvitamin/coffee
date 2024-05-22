import pandas as pd
import numpy as np
import math
import logging
from coffee import Coffee
from preprocess import granularity_resample

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - Coffee - %(message)s')


def calc_accuracy(df, error_metric='ACCURACY'):
    """
    Calculate the evaluation accuracy based on selected metric. 
    Default is 'Accuracy'

    Parameters
    ----------
    df : pandas DataFrame
        df of evaluation result
    error_metric : string
        'ACCURACY', 'MSE', 'RMSE', 'MAE'

    Returns
    -------
    df : pandas DataFrame
        df with columns "error" and "accuracy"
    """

    df['error'] = df['predict'] - df['value']
    df['accuracy'] = 1 - abs(df['error'])/df['value']

    if error_metric == 'ACCURACY':
        error_metric_value = df['accuracy'].mean()
    elif error_metric == 'MSE':
        error_metric_value = np.mean([x**2 for x in df['error']])
    elif error_metric == 'RMSE':
        error_metric_value = math.sqrt(np.mean([x**2 for x in df['error']]))
    elif error_metric == 'MAE':
        error_metric_value = np.mean([abs(x) for x in df['error']])
    else:
        logging.warning('No such error metric yet!')
        return df

    sample_size = df.shape[0]
    logging.info('Evaluation on data with size %s : %s = %s' % (sample_size, error_metric, error_metric_value))

    return df


def batch_test(coffee, df):
    """ 
    Evaluate the model in batch style. 
    For example, split the data once into training set and testing set.

    Parameters
    ----------
    coffee : obj
       Initialized coffee class
    df : pandas dataframe
       Raw data

    Returns
    -------
    forecast
        Evaluation result in pandas df format

    """
    granularity = coffee.default_params['granularity']
    df = granularity_resample(df, granularity)

    train_size_ratio = 1 - coffee.default_params['evaluate_ratio']
    train_len = int(df.shape[0] * train_size_ratio)

    logging.info('Evaluation starts from %s' % df.iloc[train_len].date)
    _, model = coffee.fit(df.iloc[:train_len + 1])
    trained_features, _ = coffee.fit(df)

    trained_features['predict_date'] = df.iloc[train_len:].date.tolist()
    forecast = coffee.predict(model, trained_features)

    return forecast


def rolling_test(coffee, df):
    """
    !!! May not work properly

    Evaluate the model in rolling style. 
    For example, rolling the batch test.

    Parameters
    ----------
    coffee : obj
       Initialized Coffee instance
    df : pandas dataframe
       Raw data

    Returns
    -------
    forecast
        Evaluation result in pandas df format
    """

    if coffee.default_params['evaluate_size']:
        train_len = int(df.shape[0] - coffee.default_params['evaluate_size'])
    else:
        train_size_ratio = 1 - coffee.default_params['evaluate_ratio']
        train_len = int(df.shape[0] * train_size_ratio)
    
    logging.info('Rolling Evaluation starts from %s' % df.iloc[train_len].date)
    forecast = pd.DataFrame()

    for i in range(train_len, len(df)):
        temp_df = df.iloc[:i]
        temp_forecast = coffee.run_model(temp_df)
        temp_forecast['model_date'] = df.iloc[i].date
        target_value = df.iloc[i + 1].value
        temp_forecast['value'] = target_value
        forecast = pd.concat([forecast, temp_forecast])

    return forecast


def evaluation_model(df, user_params):
    """
    Main function of evaluating the coffee model

    Parameters
    ----------
    df : pandas dataframe
        Raw data in df format
    user_params : dict
        User-defined parameters for coffee

    Returns
    -------
    result
        Evaluation result in df format
    """
    user_params['model_type'] = 'evaluate'
    forecast_model = Coffee(user_params)

    evaluation_type = forecast_model.default_params['evaluate_type']

    if evaluation_type == 'batch_test':
        result = batch_test(forecast_model, df)
    elif evaluation_type == 'rolling_test':
        result = rolling_test(forecast_model, df)
    else:
        raise Exception

    if 'error_metric' in user_params.keys():
        error_metric = user_params['error_metric']
    else:
        error_metric = 'ACCURACY'

    calc_accuracy(result, error_metric)

    return result
