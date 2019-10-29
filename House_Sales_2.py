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
X = training_data.drop(['SalePrice'], axis=1)
X_test = test_data.copy()
X_test_copy = X_test.copy()

# Splitting
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_train_copy = X_train.copy()
X_val_copy = X_val.copy()
X_train_copy = X_train.select_dtypes(exclude = ['object'])
X_val_copy = X_val.select_dtypes(exclude = ['object'])
object_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']
X_train_object = X_train.select_dtypes(exclude = ['int64', 'float64'])
X_val_object = X_val.select_dtypes(exclude = ['int64', 'float64'])
X_test_copy = X_test.select_dtypes(exclude = ['object'])
X_test_object = X_test.select_dtypes(exclude = ['int64', 'float64'])

# ---------------------------------------------------------------------------------------------

# Imputing missing values
from sklearn.impute import SimpleImputer
final_imputer = SimpleImputer(strategy = 'most_frequent')
X_train_imputed = pd.DataFrame(final_imputer.fit_transform(X_train_copy), columns = X_train_copy.columns)
X_val_imputed = pd.DataFrame(final_imputer.transform(X_val_copy), columns = X_val_copy.columns)
X_test_imputed = pd.DataFrame(final_imputer.fit_transform(X_test_copy), columns = X_test_copy.columns)

X_train_imputed['temp'] = 1
X_train_object['temp'] = 1
X_train_copy = pd.merge(X_train_object, X_train_imputed, on = ['temp']).reindex(X_train.index)
X_train_copy = X_train_copy.drop('temp', axis = 1)
X_train_object = X_train_object.drop('temp', axis = 1)
X_train_imputed = X_train_imputed.drop('temp', axis = 1)
X_train_copy = X_train_copy[X_train.columns]
X_train_imputed.index = X_train_copy.index

X_val_imputed['temp'] = 1
X_val_object['temp'] = 1
X_val_copy = pd.merge(X_val_object, X_val_imputed, on = ['temp']).reindex(X_val.index)
X_val_copy = X_val_copy.drop('temp', axis = 1)
X_val_object = X_val_object.drop('temp', axis = 1)
X_val_imputed = X_val_imputed.drop('temp', axis = 1)
X_val_copy = X_val_copy[X_val.columns]
X_val_imputed.index = X_val_copy.index

X_test_imputed['temp'] = 1
X_test_object['temp'] = 1
X_test_copy = pd.merge(X_test_object, X_test_imputed, on = ['temp']).reindex(X_test.index)
X_test_copy = X_test_copy.drop('temp', axis = 1)
X_test_object = X_test_object.drop('temp', axis = 1)
X_test_imputed = X_test_imputed.drop('temp', axis = 1)
X_test_copy = X_test_copy[X_test.columns]
X_test_imputed.index = X_test_copy.index

# Dropping Categorical Variables with Missing Values
cols_with_missing = [col for col in X_train_object.columns if X_train_object[col].isnull().any()]
X_train_copy = X_train_copy.drop(cols_with_missing, axis = 1)
X_val_copy = X_val_copy.drop(cols_with_missing, axis = 1)
X_test_copy = X_test_copy.drop(cols_with_missing, axis = 1)

# Updating X_train_object by removing columns containing Null values
X_train_object = X_train_object.drop(cols_with_missing, axis = 1)
X_val_object = X_val_object.drop(cols_with_missing, axis = 1)

# Finding the number of values in object columns
low_cardinality_cols = [col for col in X_train_object if X_train_object[col].nunique() < 10]
high_cardinality_cols = list(set(X_train_object)-set(low_cardinality_cols))
print('Low Cardinality Columns:', low_cardinality_cols)
print('\nHigh Cardinality Columns:', high_cardinality_cols)

# --------------------------------------------------------------------------------------------

temp = []
for i in X_train_object:
	temp.append(list(X_train_object[i].unique()))

# Using One Hot Encoder an Label Encoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(one_hot_encoder.fit_transform(X_train_copy[low_cardinality_cols]))
OH_cols_val = pd.DataFrame(one_hot_encoder.transform(X_val_copy[low_cardinality_cols]))
OH_cols_test = pd.DataFrame(one_hot_encoder.transform(X_test_copy[low_cardinality_cols]))

OH_cols_train.index = X_train_copy.index
OH_cols_val.index = X_val_copy.index
OH_cols_test.index = X_test_copy.index


OH_X_train = pd.concat([X_train_imputed, OH_cols_train], axis=1)
OH_X_val = pd.concat([X_val_imputed, OH_cols_val], axis=1)
OH_X_test = pd.concat([X_test_imputed, OH_cols_test], axis=1)
#------------------------------------------------------------------------------------------------

# Categorical Columns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

"""model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]

def score_model(model, X_t=OH_X_train, X_v=OH_X_val, y_t=y_train, y_v=y_val):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

for i in range(0, len(models)):
    mae_rft = score_model(models[i])
    print("Model %d MAE: %d" % (i+1, mae_rft))"""

def score_mae(OH_X_train, OH_X_val, y_train, y_val):
	model = RandomForestRegressor(n_estimators=1000, max_depth = 20, random_state=0)
	model.fit(OH_X_train, y_train)
	preds = model.predict(OH_X_val)
	return mean_absolute_error(y_val, preds)   
print(score_mae(OH_X_train, OH_X_val, y_train, y_val))

rf_regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
rf_regressor.fit(OH_X_train,y_train)
predictions = rf_regressor.predict(OH_X_test)

# Results
output = pd.DataFrame({'Id': X_test_copy.index,
                       'SalePrice': predictions})
output.to_csv('submission_2.csv', index=False)
