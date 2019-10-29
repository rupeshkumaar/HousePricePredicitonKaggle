# House Sales Prediction

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the Dataset
training_data = pd.read_csv('train.csv', index_col = 'Id')
test_data = pd.read_csv('test.csv', index_col = 'Id')
training_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = training_data.SalePrice
training_data.drop(['SalePrice'], axis=1, inplace=True)
X = training_data.select_dtypes(exclude = ['object'])
X_test = test_data.select_dtypes(exclude = ['object'])
X_test_copy = X_test.copy()

# Splitting
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 0)

#---------------------Part 1- Dropping Columns------------------------------------

# Dropping Columns that contain null values
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(cols_with_missing, axis = 1)
reduced_X_val = X_val.drop(cols_with_missing, axis = 1)

# Importing the Regressors
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Linear Model
regressor_1 = LinearRegression()
regressor_1.fit(reduced_X_train, y_train)
y_pred_1 = regressor_1.predict(reduced_X_val)
mae_lr = mean_absolute_error(y_val, y_pred_1)

# Decision Tree
def get_mae(max_leaf_nodes, reduced_X_train, reduced_X_val, y_train, y_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(reduced_X_train, y_train)
    preds_val = model.predict(reduced_X_val)
    mae = mean_absolute_error(y_val, preds_val)
    return(mae)
for max_leaf_nodes in [5, 50, 100, 500, 1000, 5000]:
    mae_dt1 = get_mae(max_leaf_nodes, reduced_X_train, reduced_X_val, y_train, y_val)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, mae_dt1))

regressor_2 = DecisionTreeRegressor(random_state = 0, max_leaf_nodes = 100)
regressor_2.fit(reduced_X_train, y_train)
y_pred_2 = regressor_2.predict(reduced_X_val)
mae_dt = mean_absolute_error(y_val, y_pred_2)

# Random Forest
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]

def score_model(model, X_t=reduced_X_train, X_v=reduced_X_val, y_t=y_train, y_v=y_val):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

for i in range(0, len(models)):
    mae_rft = score_model(models[i])
    print("Model %d MAE: %d" % (i+1, mae_rft))

regressor_3 = RandomForestRegressor(n_estimators=50, random_state=0)
regressor_3.fit(reduced_X_train, y_train)
y_pred_3 = regressor_3.predict(reduced_X_val)
mae_rf = mean_absolute_error(y_val, y_pred_3)

"""The Lowest Mean Absolute Error was given by the Random Forest model_1"""

# Full Predictions
#null_columns = X_test.columns[X_test.isnull().any()]
#print(X_test[X_test['BsmtFinSF1'].isnull()][null_columns])
#reduced_X_test.isnull().any()
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'mean')
reduced_X_test = X_test_copy.drop(cols_with_missing, axis = 1)
reduced_X_test = pd.DataFrame(imputer.fit_transform(reduced_X_test))

regressor_3.fit(reduced_X_train, y_train)
rf_pred = regressor_3.predict(reduced_X_test)

#-----------------------Part 2--- Using Imputer on all the columns that has missing values-----------------------------------------------

# Imputation
final_imputer = SimpleImputer(strategy = 'median') # I tried mean strategy it gave MAE = 17606.362 and median gave MAE = 17556.745
X_train_imputed = pd.DataFrame(final_imputer.fit_transform(X_train))
X_val_imputed = pd.DataFrame(final_imputer.transform(X_val))
X_test_imputed = pd.DataFrame(final_imputer.fit_transform(X_test_copy))
"""def score_dataset(X_train, X_val, y_train, y_val):
    model = RandomForestRegressor(n_estimators=1000, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return mean_absolute_error(y_val, preds)
print(score_dataset(X_train_imputed, X_val_imputed, y_train, y_val))"""

# Applying imputation only on the Random Forest Model

final_regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
final_regressor.fit(X_train_imputed, y_train)
final_pred = final_regressor.predict(X_test_imputed)

# Results
output = pd.DataFrame({'Id': X_test_copy.index,
                       'SalePrice': final_pred})
output.to_csv('submission_1.csv', index=False)