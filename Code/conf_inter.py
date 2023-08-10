import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from sklearn.model_selection import train_test_split

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('m2.csv')

# Split the data into input (X) and output (y) variables
X = data[['acc', 'consistency']]
y = data['wpm']

# Data scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Calculate mean and standard deviation for training and testing predictions
mean_train = np.mean(y_train)
std_dev_train = np.std(y_train)
mean_test = np.mean(y_test)
std_dev_test = np.std(y_test)

# Generate x values for the normal distribution plot
x_values = np.linspace(y.min(), y.max(), 1000)

# Construct the normal distribution curve for training and testing predictions using pdf
y_values_train = norm.pdf(x_values, mean_train, std_dev_train)
y_values_test = norm.pdf(x_values, mean_test, std_dev_test)

# Calculate the 95% confidence interval for training and testing predictions
confidence_level = 0.95
z_score = norm.ppf((1 + confidence_level) / 2)
ci_lower_train = mean_train - z_score * std_dev_train
ci_upper_train = mean_train + z_score * std_dev_train
ci_lower_test = mean_test - z_score * std_dev_test
ci_upper_test = mean_test + z_score * std_dev_test

# Plot the histogram for training and testing predictions
plt.hist(y_train, bins=20, density=True, color='blue', alpha=0.7, label='Train Predictions')
plt.hist(y_test, bins=20, density=True, color='red', alpha=0.7, label='Test Predictions')

# Plot the normal distribution curve for training and testing predictions
plt.plot(x_values, y_values_train, color='blue', label='Train Normal Distribution')
plt.plot(x_values, y_values_test, color='red', label='Test Normal Distribution')

# Plot the 95% confidence interval for training and testing predictions
plt.axvline(ci_lower_train, color='blue', linestyle='dashed', label='Train 95% CI')
plt.axvline(ci_upper_train, color='blue', linestyle='dashed')
plt.axvline(ci_lower_test, color='red', linestyle='dashed', label='Test 95% CI')
plt.axvline(ci_upper_test, color='red', linestyle='dashed')

# Set plot labels and title
plt.xlabel('Predicted wpm')
plt.ylabel('Density')
plt.title('Normal Distribution of Predicted wpm with 95% Confidence Interval')
plt.legend()

# Display the plot
plt.show()
