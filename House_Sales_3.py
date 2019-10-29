# House Sales Prediction

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
training_data = pd.read_csv('train.csv', index_col = 'Id')
test_data = pd.read_csv('test.csv', index_col = 'Id')
training_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = training_data.SalePrice
X = training_data.drop(['SalePrice'], axis=1)
X_test_copy = test_data.copy()

# Splitting
from sklearn.model_selection import train_test_split
X_train_copy, X_val_copy, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Low Cardinality Columns
low_cardinality_cols = [col for col in X_train_copy.columns if X_train_copy[col].nunique() < 10 and 
						   X_train_copy[col].dtype == 'object']

numeric_cols = [col for col in X_train_copy.columns if X_train_copy[col].dtype in ['int64', 'float64']]

my_cols = low_cardinality_cols + numeric_cols

X_train = X_train_copy[my_cols].copy()
X_val = X_val_copy[my_cols].copy()
X_test = X_test_copy[my_cols].copy()

# One-hot encode the data (to shorten the code, we use pandas)
X_train = pd.get_dummies(X_train)
X_val = pd.get_dummies(X_val)
X_test = pd.get_dummies(X_test)
X = pd.get_dummies(X)
X_train, X_val = X_train.align(X_val, join='left', axis=1)
X, X_test = X.align(X_test, join='left', axis=1)

# XGBOOST prediction
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
regressor = XGBRegressor(n_estimators = 451, learning_rate = 0.1)
regressor.fit(X_train, y_train)
pred = regressor.predict(X_val)
score = mean_absolute_error(y_val, pred)
print(score)

# Now Predictiong for full dataset
final_regressor = XGBRegressor(n_estimators = 451, learning_rate = 0.1)
final_regressor.fit(X, y)
predictions = final_regressor.predict(X_test)

# Results
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': predictions})
output.to_csv('submission_3.csv', index=False)

