# House Sales Prediction

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the Dataset
training_data = pd.read_csv('train.csv', index_col = 'Id')
test_data = pd.read_csv('test.csv', index_col = 'Id')
features = ['MSSubClass', 'LotArea', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
          'BedroomAbvGr', 'TotRmsAbvGrd', 'MoSold', 'YrSold']
X = training_data[features]
y = training_data.SalePrice
X_test_full = test_data[features].copy()

# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting the algorithms
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Linear Model

regressor_1 = LinearRegression()
regressor_1.fit(X_train, y_train)
y_pred_1 = regressor_1.predict(X_test)
mae_lr = mean_absolute_error(y_test, y_pred_1)


# Decision Tree Regression

def get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(X_train, y_train)
    preds_val = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds_val)
    return(mae)
for max_leaf_nodes in [5, 50, 100, 500, 1000, 5000]:
    mae_dt = get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, mae_dt))

regressor_2 = DecisionTreeRegressor(random_state = 0, max_leaf_nodes = 100)
regressor_2.fit(X, y)
y_pred_2 = regressor_2.predict(X_test_full)

# Random Forest

model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]

def score_model(model, X_t=X_train, X_v=X_test, y_t=y_train, y_v=y_test):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

for i in range(0, len(models)):
    mae_rf = score_model(models[i])
    print("Model %d MAE: %d" % (i+1, mae_rf))

regressor_3 = RandomForestRegressor(n_estimators = 100, criterion = 'mae', random_state = 0)
regressor_3.fit(X, y)
y_pred_3 = regressor_3.predict(X_test_full)

best_model = y_pred_3 #Lowest MAE among all the models

output = pd.DataFrame({'Id': X_test_full.index,
                       'SalePrice': y_pred_3})
output.to_csv('submission.csv', index=False)