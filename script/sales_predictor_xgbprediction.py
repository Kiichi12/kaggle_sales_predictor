import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
import pickle
import os

#loading datasets
sales = pd.read_csv("test.csv")
holidays = pd.read_csv("holidays_events.csv")
transactions = pd.read_csv("transactions.csv")
oil = pd.read_csv("oil.csv")
stores = pd.read_csv("stores.csv")

#preprocessing dates
for df in [sales,holidays,transactions,oil]:
    df['date'] = pd.to_datetime(df['date'])

#joining the tables into one for training
data = sales.merge(stores, on='store_nbr', how='left')
data = data.merge(transactions, on=['date', 'store_nbr'], how='left')
data = data.merge(oil, on='date', how='left')
data = data.merge(holidays, on='date', how='left')

#date features
data['day'] = data['date'].dt.day
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year
data['weekday'] = data['date'].dt.weekday
data['is_weekend'] = data['weekday'].isin([5,6]).astype(int)

#encode categorical variables
label_encoders = {}
categorical_columns = ['family', 'type_x', 'type_y', 'locale', 'locale_name', 'description', 'city', 'state']

for col in categorical_columns:
    le=LabelEncoder()
    data[col] = data[col].fillna('Unknown')    #replace missing values with 'unknown'

    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le    #for future:- conversions and inverse conversions

#all remaining missing values are replaced with zeroes
data.fillna(0, inplace=True)

#relevant features for training
features = data.drop(columns=['id', 'date'], errors='ignore')

#scaling all data for uniformity
scaler = MinMaxScaler(feature_range=(0, 10))
features_scaled = scaler.fit_transform(features)

#load model
model_path = os.path.join(os.getcwd(),"xgb_predictor.json")

model = xgb.XGBRegressor()
model.load_model(model_path)  

#predict on test data
predictions = model.predict(features_scaled)

#ensure no negative sales values
predictions = np.maximum(predictions, 0)  #replace negatives with zero

#create submission file
submission = pd.DataFrame({
    'id': data['id'],
    'sales': predictions
})

#save submission CSV
submission.to_csv('xgb_sales_predictions.csv', index=False)

print("Submission file created successfully!")
