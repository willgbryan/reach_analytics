import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score
import logging
import numpy as np
import datetime
from pandas.api.types import is_datetime64_ns_dtype
import gc


import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score
import logging
import numpy as np
import datetime
from pandas.api.types import is_datetime64_ns_dtype
import gc


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def feature_engineering(series_path):
    logging.info("Reducing memory impact...")
    """ 
    Iterate through all numeric columns of a dataframe and modify the data type
    to reduce memory usage.        
    """

    df = pd.read_parquet(series_path)
    
    start_mem = df.memory_usage().sum() / 1024**2
    logging.info(f'Memory usage of dataframe is {start_mem:.2f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and not is_datetime64_ns_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        
    df['series_id'] = df['series_id'].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    logging.info(f'Memory usage after optimization is: {end_mem:.2f} MB')
    decrease = 100 * (start_mem - end_mem) / start_mem
    logging.info(f'Decreased by {decrease:.2f}%')
    
    logging.info("Performing feature engineering...")
    df['timestamp'] = pd.to_datetime(df['timestamp']).apply(lambda t: t.tz_localize(None))
    df.sort_values(['series_id', 'timestamp'], inplace=True)
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['second'] = df['timestamp'].dt.second
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['anglez_rolling_avg'] = df['anglez'].rolling(window=10).mean()
    df['anglez_diff'] = df['anglez'].diff()
    df['enmo_rolling_avg'] = df['enmo'].rolling(window=10).mean()
    df['enmo_diff'] = df['enmo'].diff()
    df['anglez_x_enmo'] = df['anglez'] * df['enmo']
    df['anglezabs_x_enmo'] = abs(df['anglez']) * df['enmo']
    df['relu_anglez_x_enmo'] = np.maximum(df['anglez'], 0.) * df['enmo']
#     for n in range(1, 10):
#             df[f'{col}_lag_{n}'] = df.groupby('series_id')[col].shift(n)
#     gc.collect()
#     logging.info(f'Created lag derivates from {col}')
    
#     for n in range(1, 10):
#         df[f'{col}_lead_{n}'] = df.groupby('series_id')[col].shift(-n)
#     gc.collect() 
#     logging.info(f'Created lead derivates from {col}')
    
#     # circle back to here
#     for n in range(1, 10):
#         df[f'{col}_lead_{n}'] = df[f'{col}_lead_{n}'].fillna(0)
    columns_to_fill = ['anglez', 'enmo', 'anglez_rolling_avg', 'anglez_diff', 'enmo_rolling_avg', 'enmo_diff', 'anglez_x_enmo', 'anglezabs_x_enmo', 'relu_anglez_x_enmo']
    df[columns_to_fill] = df[columns_to_fill].fillna(0)

    return df

def preprocess_data(engineered_df, event_path=None, subset_fraction=1.0):
    logging.info(f"Preprocessing data from {series_path}...")

    series_df = engineered_df
    
    # Subsetting series_df
    if subset_fraction < 1.0:
        series_df = series_df.sample(frac=subset_fraction).reset_index(drop=True)

    if event_path:
        events_df = pd.read_csv(event_path)
        events_df['timestamp'] = pd.to_datetime(events_df['timestamp']).apply(lambda t: t.tz_localize(None))

        if subset_fraction < 1.0:
            events_df = events_df.sample(frac=subset_fraction).reset_index(drop=True)
        # Merge series with events
        series_df = series_df.merge(events_df, how='left', on=['series_id', 'timestamp', 'step'])
        # One-Hot Encoding for 'event' column
        events_ohe = pd.get_dummies(series_df["event"], prefix="event")
        events_ohe['event_awake'] = series_df['awake']
        series_df = pd.concat([series_df, events_ohe], axis=1)
        series_df.drop("event", axis=1, inplace=True) # Drop original event column

        event_to_int = {col: idx for idx, col in enumerate(events_ohe.columns)}

        return series_df, event_to_int # Return one-hot encoded column names

    return series_df, None

# Preprocessing training data
logging.info("Starting data preprocessing for training data...")
series_path = 'C:/Users/willb/Downloads/train_series.parquet'
event_path = 'cmi_sleep_states/train_events.csv'

subset_percentage = 0.1
train_data, event_to_int = preprocess_data(feature_engineering(series_path), event_path, subset_percentage)
event_df = pd.read_csv('cmi_sleep_states/train_events.csv')

# Assign features for training data
X_train = train_data.drop(['series_id', 'timestamp', 'night'] + list(event_to_int.keys()), axis=1)

# Transform y_train to integer format based on position of '1' in the one-hot encoded representation
y_train = train_data[list(event_to_int.keys())].idxmax(axis=1).map(event_to_int)

# Preprocess the test data
logging.info("Starting data preprocessing for test data...")
test_series_path = 'cmi_sleep_states/test_series.parquet'
X_test, _ = preprocess_data(feature_engineering(test_series_path))
test_series_id = X_test['series_id'].copy()
test_timestamp = X_test['timestamp'].copy()
X_test = X_test.drop(['series_id', 'timestamp'], axis=1)

# Ensure the test data has the same columns as training data
if set(X_train.columns) != set(X_test.columns):
    logging.warning("Mismatch in features between training and test set!")
    logging.info(X_train.columns)
    logging.info(X_test.columns)
    missing_columns = set(X_train.columns) - set(X_test.columns)
    additional_columns = set(X_test.columns) - set(X_train.columns)
    logging.warning(f"Missing columns in test set: {missing_columns}")
    logging.warning(f"Additional columns in test set: {additional_columns}")

# Initialize and train the 
logging.info("Initializing and training the model...")
model = xgb.XGBClassifier(objective='multi:softprob', num_class=len(event_to_int), n_jobs=-1)
model.fit(X_train, y_train)

# Predict the probabilities on test series
pred_probs = model.predict_proba(X_test)

# Mapping from encoded label to original label
int_to_event = {v: k for k, v in event_to_int.items()}

# Create a DataFrame to hold the results
results = []
for i, series_id in enumerate(test_series_id):
    for j, class_prob in enumerate(pred_probs[i]):
        results.append({
            'row_id': len(results),
            'series_id': series_id,
            'step': i,
            'event': int_to_event[j],
            'score': class_prob
        })

result_df = pd.DataFrame(results)
# Map event names
event_mapping = {
    'event_onset': 'onset',
    'event_wakeup': 'wakeup'
}
result_df['event'] = result_df['event'].map(event_mapping).fillna(result_df['event'])

# Remove rows with 'event_awake'
result_df = result_df[result_df['event'] != 'event_awake']

# Save the predictions in the desired format
result_df.to_csv('submission.csv', index=False)


