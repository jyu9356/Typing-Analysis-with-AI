import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('w1050.csv')

# Split the data into input (X) and output (y) variables
X = data[['acc', 'consistency']]
y = data['wpm']

# Fit a linear regression model
regression_model = LinearRegression()
regression_model.fit(X, y)

# Get the predicted values
predicted_values = regression_model.predict(X)

# Compute the mean and standard deviation of the 'wpm' variable
wpm_mean = y.mean()
wpm_std = y.std()

# Print the mean and standard deviation
print("Mean of 'wpm' variable:", wpm_mean)
print("Standard Deviation of 'wpm' variable:", wpm_std)

# Generate the x-values for the bell-shaped curve
#x = np.linspace(predicted_values.min(), predicted_values.max(), 100)
x = np.linspace(wpm_mean - 3*wpm_std, wpm_mean + 3*wpm_std, 100)

# Calculate the corresponding y values using the probability density function (PDF)
y = norm.pdf(x, wpm_mean, wpm_std)

# Plot the bell-shaped curve
plt.plot(x, y, color='blue')

# Set plot labels and title
plt.xlabel('Predicted Values')
plt.ylabel('Frequency')
plt.title('Histogram of Predicted Values with Bell-Shaped Curve')

# Display the plot
plt.show()
