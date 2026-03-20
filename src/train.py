import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Loading data jan + feb
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

# Feature engineering
data['store_and_fwd_flag'] = (data['store_and_fwd_flag'] == 'Y').astype(int)

# Target and features
reg_features = ['trip_distance', 'tolls_amount', 'PULocationID', 'VendorID']
X = data[reg_features]
y = data['fare_amount']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
reg_model = RandomForestRegressor(n_estimators=10, random_state=42)
reg_model.fit(X_train, y_train)

# Evaluation
y_pred = reg_model.predict(X_test)
print("V3 Training Results (Jan 2021 + Feb 2021):")
print(f"MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R2:   {r2_score(y_test, y_pred):.4f}")

# Saving the model
joblib.dump(reg_model, 'models/random_forest_regression.pkl')
print("Model saved!")