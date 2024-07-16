from linreg import LinearRegression
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
#load dataset
df = pandas.read_csv('advertising.csv')
linreg = LinearRegression()

#prepare data
X = df['Newspaper'].values
y = df['Sales'].values
X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1, 1), y.reshape(-1, 1), test_size = 0.2, random_state=0)
linreg.fit(X_train, y_train)

#make predictions
predictions = linreg.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(predictions)

