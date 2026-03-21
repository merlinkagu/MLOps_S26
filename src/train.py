import pandas as pd
import numpy as np
import joblib
import yaml
from dvclive import Live
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def main():
    params = load_params('params.yaml')
    test_size = params['train']['test_size']
    random_state = params['train']['random_state']
    n_estimators = params['train']['n_estimators']
    outlier_distance = params['data']['outlier_distance']

    # Loading data
    df_jan = pd.read_parquet(params['data']['train_path'])
    df_feb = pd.read_parquet(params['data']['test_path'])
    data = pd.concat([df_jan, df_feb], ignore_index=True)

    # Basic filtering
    data = data.drop(columns=['ehail_fee'])
    data = data[
        (data['fare_amount'] > 0) &
        (data['total_amount'] > 0) &
        (data['mta_tax'] >= 0) &
        (data['extra'] >= 0) &
        (data['tip_amount'] >= 0) &
        (data['trip_distance'] > 0) &
        (data['lpep_dropoff_datetime'] > data['lpep_pickup_datetime'])
    ]

    # Outlier removal
    data.loc[data['trip_distance'] > outlier_distance, 'trip_distance'] = np.nan
    data.loc[data['passenger_count'] == 0, 'passenger_count'] = np.nan

    # Imputation
    data['trip_distance'] = data['trip_distance'].fillna(data['trip_distance'].median())
    median_imputer = SimpleImputer(strategy='median')
    data[['passenger_count']] = median_imputer.fit_transform(data[['passenger_count']])
    freq_imputer = SimpleImputer(strategy='most_frequent')
    data[['RatecodeID','payment_type','trip_type','congestion_surcharge']] = freq_imputer.fit_transform(
        data[['RatecodeID','payment_type','trip_type','congestion_surcharge']]
    )

    # Target and features
    reg_features = ['trip_distance', 'tolls_amount', 'PULocationID', 'VendorID']
    X = data[reg_features]
    y = data['fare_amount']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Training
    reg_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    reg_model.fit(X_train, y_train)

    # Evaluation
    y_pred = reg_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

    # dvclive logging
    with Live(save_dvc_exp=True) as live:
        live.log_metric('mae', mae)
        live.log_metric('rmse', rmse)
        live.log_metric('r2', r2)
        live.log_params(params)

    # Saving the model
    joblib.dump(reg_model, 'models/random_forest_regression.pkl')
    print("Model saved!")

if __name__ == '__main__':
    main()