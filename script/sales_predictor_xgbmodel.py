import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import load_model
import pickle

#loading datasets
sales = pd.read_csv("train.csv")
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
    data[col] = data[col].fillna('Unknown')    #replace missing values with 'Unknown'

    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le    #for future:- conversions and inverse conversions

#all remaining missing values are replaced with zeroes
data.fillna(0, inplace=True)

#relevant features for training
features = data.drop(columns=['sales', 'id', 'date'], errors='ignore')
target = data['sales']

#scaling all data for uniformity
scaler = MinMaxScaler(feature_range=(0, 10))
features_scaled = scaler.fit_transform(features)

model=xgb.XGBRegressor(objective='reg:squarederror', random_sate=12)

#possible parameters of the regressor model
param_grid = {
    'n_estimators': [100, 300, 500],        #number of trees
    'learning_rate': [0.01, 0.05, 0.1],     #step size shrinkage
    'max_depth': [6, 7, 9],                 #depth of each tree
    'subsample': [0.7, 0.8, 0.9],           #fraction of samples used per tree
    'colsample_bytree': [0.7, 0.8, 0.9],    #fraction of features used per tree
    'gamma': [0, 0.1, 3],                   #minimum loss reduction to make a split
    'lambda': [1, 3, 5]                     #L2 regularization
}

#parameters of grid search
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',   #can use other metrics like 'r2' or 'neg_mean_absolute_error'
    cv=3,                               #3-fold cross-validation
    verbose=2,
    n_jobs=-1                           #use all available CPU cores
)

grid_search.fit(features_scaled, target)  #try all possible combinations

#the best parameters
print("Best Parameters:", grid_search.best_params_)

#train the model
best_model = xgb.XGBRegressor(**grid_search.best_params_)
best_model.fit(features_scaled, target)

#save the model
best_model.save_model("xgb_predictor.json")

print("Model saved successfully!")


