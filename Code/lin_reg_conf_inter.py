import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

data = pd.read_csv('m2.csv')

X_accuracy = data[['acc']]
y = data['wpm']

# Split data into train and test sets
X_accuracy_train, X_accuracy_test, y_train, y_test = train_test_split(X_accuracy, y, test_size=0.2, random_state=42)

# Create linear regression model
accuracy_model = LinearRegression()

# Fit the model
accuracy_model.fit(X_accuracy_train, y_train)

# Predict y values for test set
y_accuracy_pred = accuracy_model.predict(X_accuracy_test)

# Calculate the mean squared error for the model
mse_accuracy = mean_squared_error(y_test, y_accuracy_pred)

# Calculate the standard error of the estimate (standard deviation of the residuals)
standard_error = np.sqrt(mse_accuracy)

# Calculate the 95% confidence intervals for each x value
# The confidence interval will be ± 1.96 * standard error
confidence_interval = 1.96 * standard_error

# Plot the data points and the regression line
plt.scatter(X_accuracy_test, y_test, label='Actual', alpha=0.7)
plt.plot(X_accuracy_test, y_accuracy_pred, color='red', label='Predicted')
plt.fill_between(X_accuracy_test.squeeze(), (y_accuracy_pred - confidence_interval).squeeze(), (y_accuracy_pred + confidence_interval).squeeze(), color='gray', alpha=0.3, label='95% Confidence Interval')
plt.xlabel('Accuracy')
plt.ylabel('Words per Minute (WPM)')
plt.legend()
plt.title('Linear Regression with 95% Confidence Interval')
plt.show()

# Calculate the upper and lower bounds of the confidence interval
upper_bound = y_accuracy_pred + confidence_interval
lower_bound = y_accuracy_pred - confidence_interval

# Count the number of predicted values within the confidence interval
within_interval = np.logical_and(y_test >= lower_bound, y_test <= upper_bound)
fraction_within_interval = np.sum(within_interval) / len(y_test)

print("Fraction of points within the 95% confidence interval:", fraction_within_interval)
