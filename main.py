import pandas as pd
import pickle
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Read the data
data = pd.read_csv('Electric Load Data.csv')

# Fit the data to an ML model
x = data[['L(T-24)', 'L(T-48)', 'L(T-72)', 'L(T-168)', 'L(T-336)', 'L(T-504)']]
y = data['L(T)']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

model = RandomForestRegressor(n_estimators=20)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Save the model
filename = 'model/model.pkl'
pickle.dump(model, open(filename, 'wb'))
