import numpy as np
import pandas as pd
from datetime import timedelta
from dateutil.parser import parse
from scipy import sparse
import logging
from multiprocessing import Pool
# from chinese_calendar import is_holiday


def feature_transform(df, default_params):
    """
    General Feature Transform Function

    Parameters
    ----------
    df : pandas DataFrame
        Raw data
    default_params : dict
        Coffee params

    Returns
    -------
    date : list
        List of value dates
    feature : list
        List of features
    label : list
        List of labels
    """

    window_size = default_params['window_size']
    predict_len = default_params['predict_len']
    backtrace_step = default_params['backtrace_step']

    date_arr = df['date'].values

    if default_params['target_preprocessing'] == "log1p":
        target_arr = np.log1p(df['value'].values)
    else:
        target_arr = df['value'].values

    feature = []
    label = []
    date = []

    for i in range(len(date_arr) - 1, window_size * backtrace_step + predict_len, -backtrace_step):
        label.append(target_arr[i])
        date.append(date_arr[i])
        feature_ls = []
        # Matching corresponing external feature based on date
        if default_params['external_feature']:
            additional_feature = default_params['external_feature'][date_arr[i]]
            feature_ls.extend(additional_feature)
        feature_ls.extend(target_arr[i - window_size - predict_len + 1: i - predict_len + 1])
        feature.append(np.array(feature_ls))

    return date, feature, label


def generate_similar_result(similarity_feature, similarity_type, backtrace_days):
    """
    Function for finding similar dates in history

    Parameters
    ----------
    similarity_feature : dict
        Passed in from params_dict, {datetime: [v1, v2, ... ]}
    similarity_type : str
        Defined in params_dict, 'euclidean', 'cosine' or 'dtw'
    backtrace_days : int
        Choose top-{backtrace_days} dates

    Returns
    -------
    similar_dict : dict
        List of dates, sorted by similarity

    """
    date_sorted = sorted(similarity_feature.keys(), reverse=True)
    similarity_array = np.array([list(similarity_feature[date]) for date in date_sorted])

    # Generate feature matrix
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
    similarity_feature_sparse = sparse.csr_matrix(similarity_array)

    # Calculate similarity and get indexes
    if similarity_type == "cosine":
        similarity_result = cosine_similarity(similarity_feature_sparse)
        order_matrix = similarity_result.argsort()[::-1]
    elif similarity_type == "euclidean":
        similarity_result = euclidean_distances(similarity_feature_sparse)
        order_matrix = similarity_result.argsort()
    elif similarity_type == "dtw":
        similarity_result = dtw_distance_mx(similarity_array)
        order_matrix = similarity_result.argsort()
    else:
        raise ValueError("Unknown similarity type")
    
    similar_dict = {}
    for row_id in range(len(order_matrix)):
        candidates = [x for x in order_matrix[row_id] if x > row_id][:backtrace_days]
        result = []
        for index in candidates:
            result.append((index, similarity_result[row_id][index]))
        similar_dict[row_id] = result

    return similar_dict


def pairwise_transform_similarity(date, feature, label, default_params):
    """
    Pairwise transform of base features with similarity search

    Parameters
    ----------
    date : list
        List of data date in format 'yyyy-MM-dd'
    feature : list
        List of base features
    label : list
        List of labels
    default_params : dict
        default_params in coffee

    Returns
    -------
    pairwise_feature : list
        Features after pairwise
    pairwise_label : list
        Labels after pairwise
    pairwise_dates : list
        List of pairwise dates. For example, [(date1, date2),...]
    pairwise_value : list
        List of original value of pairwise dates.
    pairwise_similarity : list
        List of similarity score between two dates
    """
    pairwise_dates = []
    pairwise_feature = []
    pairwise_label = []
    pairwise_value = []
    pairwise_similarity = []

    backtrace_days = default_params['backtrace_days']
    similarity_feature = default_params['similarity_feature']
    similarity_type = default_params['similarity_type']

    if similarity_feature is None:
        raise ValueError("Not specify similarity feature")

    similarity_feature = {k: v for k, v in similarity_feature.items() if k in date}
    similar_dic = generate_similar_result(similarity_feature, similarity_type, backtrace_days)

    for i in range(0, len(feature) - backtrace_days):
        ls = similar_dic[i]
        for pair in ls:
            index = pair[0]
            similarity = pair[1]
            left_date = date[index]
            right_date = date[i]
            pairwise_dates.append([left_date, right_date])
            pairwise_label.append(label[i] - label[index])
            base_feature = feature[i] - feature[index]
            extend_feature = []
            extend_feature.extend(generate_date_feature(left_date))
            extend_feature.extend(generate_date_feature(right_date))
            pairwise_feature.append(np.concatenate([base_feature, extend_feature]))
            pairwise_value.append([label[index], label[i]])
            pairwise_similarity.append(similarity)

    return pairwise_feature, pairwise_label, pairwise_dates, pairwise_value, pairwise_similarity


def pairwise_transform(date, feature, label, default_params):
    """ 
    Pairwise transform of base features in search of previous dates

    Parameters
    ----------
    date : list
        List of data date in format 'yyyy-MM-dd'
    feature : list
        List of base features
    label : list
        List of labels
    default_params : dict
        default_params in coffee

    Returns
    -------
    pairwise_feature : list
        Features after pairwise
    pairwise_label : list
        Labels after pairwise
    pairwise_dates :list
        List of pairwise dates. For example, [(date1, date2),...]
    pairwise_value
        List of original value of pairwise dates.
    """

    pairwise_dates = []
    pairwise_feature = []
    pairwise_label = []
    pairwise_value = []
    backtrace_days = default_params['backtrace_days']

    for i in range(0, len(feature) - backtrace_days):
        for j in range(1, backtrace_days):
            left_date = date[i + j]
            right_date = date[i]
            pairwise_dates.append([left_date, right_date])
            pairwise_label.append(label[i] - label[i + j])
            base_feature = feature[i] - feature[i + j]
            pairwise_feature.append(np.concatenate([base_feature]))
            pairwise_value.append([label[i + j], label[i]])

    return pairwise_feature, pairwise_label, pairwise_dates, pairwise_value


def get_predict_date(df, predict_len, granularity):
    """ Generate the predict date

    Parameters
    ----------
    df : pandas DataFrame
        Raw data
    predict_len : int
        Length of prediction
    granularity : str
        Granularity of prediction
    Returns
    -------
    predict_date : list
        Date of prediction

    """

    predict_date = []
    max_date = df.date.max()
    if granularity == '1/24':
        predict_date.append((max_date + timedelta(days=predict_len)).strftime('%Y-%m-%d'))
    elif granularity == '1':
        predict_date.append((max_date + timedelta(hours=predict_len)).strftime("%Y-%m-%d %H:%M:%S"))
    elif granularity == '1/168':
        predict_date.append((max_date + timedelta(weeks=predict_len)).strftime("%Y-%m-%d"))
    else:
        return False

    return predict_date


def generate_date_feature(date):
    """ Generate day feature such as day_of_week and is_holiday

    Parameters
    ----------
    date : pandas.Datetime
        Date of target entry

    Returns
    ----------
    A list of day feature

    """
    date_time = pd.to_datetime(date)
    dow = date_time.dayofweek
    # holiday = int(is_holiday(date_time))
    return [dow]


def granularity_resample(df, granularity):
    """ 
    Granularity resample of the Dataframe

    Parameters
    ----------
    df : pandas.Dataframe
        Original dataframe
    granularity : str
        Granularity to transform such as '1/24', '1', '1/168'

    Returns
    ----------
    df : pandas.Dataframe
        Transformed dataframe
    """

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').copy()
    df = df.set_index("date")
    if granularity == '1/24':
        try:
            df = df.resample("1D").sum()
        except ValueError:
            logging.error('Please check the granularity of your df.')
    elif granularity == '1':
        try:
            df = df.resample("1H").sum()
        except ValueError:
            logging.error('Please check the granularity of your df.')
    elif granularity == '1/168':
        try:
            df = df[len(df) % 7:]
            df = df.resample("7D").sum()
        except ValueError:
            logging.error('Please check the granularity of your df.')
    else:
        raise ValueError("Un-supported granularity. Pick one in ['1/24', '1', '1/168']")
    df = df.reset_index()

    return df

def external_feature_transform_helper(df, preprocessing_dict, granularity, predict_len):
    """ 
    Helper function to generate external features.

    Parameters
    ----------
    df : pandas dataframe
        Raw data
    preprocessing_dict :  dict
        In format such as {'column to handle': [lag days, actual function to transform]}.
        For example: {'DAU':[4, log1p]}
    granularity : str
        Granularity of the data: '1/24', '1', '1/168'
    predict_len : int
        The length of prediction
    Returns
    ----------
    external_feature : dict
        A big dictionary in format such as {date1: numpy array of external features of date1,
                                            date2: nnumpy array of external features of date2,
                                            ...}

    """

    df = granularity_resample(df, granularity)

    predict_date = get_predict_date(df, predict_len, granularity)
    predict_df = pd.DataFrame({"date": [parse(x).strftime('%Y-%m-%d %H:%M:%S') for x in predict_date]})
    df = pd.concat([df, predict_df])

    column_list = list(preprocessing_dict.keys())
    column_list.append('date')
    df_external = df[column_list].copy()
    df_external['date'] = pd.to_datetime(df_external["date"])
    df_external = df_external.sort_values("date")
    max_lagdays = 0
    for key in preprocessing_dict.keys():
        max_lagdays = max(max_lagdays, preprocessing_dict[key][0])

    external_feature = {}
    date_arr = np.array(df_external['date'])
    for i in range(max_lagdays + predict_len, len(date_arr)):
        feature_ls = []
        for j in range(len(column_list)):
            if column_list[j] != "date":
                external_feature_arr = np.array(df[column_list[j]])
                lagdays = preprocessing_dict[column_list[j]][0]
                raw_arr = external_feature_arr[i - lagdays - predict_len: i - predict_len]
                feature_arr = preprocessing_dict[column_list[j]][1](raw_arr)
                feature_ls.extend(feature_arr)
        external_feature[np.datetime64(date_arr[i])] = feature_ls
    return external_feature


def dtw_distance(s1, s2):
    """ 
    Calculate dtw distance between two vectors

    Parameters
    ----------
    s1 : numpy array
        Vector1 to compare with
    s2 : numpy array
        Vector2 to compare with

    Returns
    ----------
    dtw distance : numpy value

    """
    dtw = {}
    for i in range(len(s1)):
        dtw[(i, -1)] = float('inf')
    for j in range(len(s2)):
        dtw[(-1, j)] = float('inf')

    dtw[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = (s1[i] - s2[j]) ** 2
            dtw[(i, j)] = dist + min(dtw[(i - 1, j)], dtw[(i, j - 1)], dtw[(i - 1, j - 1)])

    return np.sqrt(dtw[len(s1) - 1, len(s2) - 1])


def dtw_distance_mx(all_ts):
    """ 
    Generate a triangular matrix of dtw distance from a array of multi vectors

    Parameters
    ----------
    all_ts : numpy array
        List of many vectors
    Returns
    ----------
    A N*N triangular matrix of dtw distance between any of two vectors in numpy

    """
    
    result = []
    p = Pool()
    for i in range(len(all_ts)):
        left_ts = all_ts[i]
        sub_result = []
        for right_ts in all_ts[i:]:
            sub_result.append(p.apply_async(dtw_distance, args=(left_ts, right_ts)))
        result.append(sub_result)
    p.close()
    p.join()
    result = [[row.get() for row in sub_result] for sub_result in result]
    arr_len = len(result)
    mat = np.zeros((arr_len, arr_len))
    for i in range(len(result)):
        for j in range(len(result[i])):
            mat[i, i + j] = result[i][j]
    return mat + mat.T - np.diag(np.diag(mat))
