import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Loading  jan + feb data
df_jan = pd.read_parquet('data/green_tripdata_2021-01.parquet')
df_feb = pd.read_parquet('data/green_tripdata_2021-02.parquet')
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
data.loc[data['trip_distance'] > 150, 'trip_distance'] = np.nan
data.loc[data['passenger_count'] == 0, 'passenger_count'] = np.nan

# Imputation
data['trip_distance'] = data['trip_distance'].fillna(data['trip_distance'].median())
median_imputer = SimpleImputer(strategy='median')
data[['passenger_count']] = median_imputer.fit_transform(data[['passenger_count']])
freq_imputer = SimpleImputer(strategy='most_frequent')
data[['RatecodeID','payment_type','trip_type','congestion_surcharge']] = freq_imputer.fit_transform(
    data[['RatecodeID','payment_type','trip_type','congestion_surcharge']]
)

# Features and target
reg_features = ['trip_distance', 'tolls_amount', 'PULocationID', 'VendorID']
X = data[reg_features]
y = data['fare_amount']

# Loading v1 model (trained only on jan data)
model = joblib.load('models/random_forest_regression.pkl')

# Evaluation
y_pred = model.predict(X)
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print("V2 Evaluation: Jan model on Jan+Feb data:")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")