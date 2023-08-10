import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
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

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions using the trained model
predictions_train = model.predict(X_train)
predictions_test = model.predict(X_test)

# Print the coefficient values
coefficients = model.coef_
intercept = model.intercept_
for i, name in enumerate(X.columns):
    print(f"Coefficient for {name}: {coefficients[i]}")
print(f"Intercept: {intercept}")

# Calculate mean squared error for training and testing sets
mse_train = mean_squared_error(y_train, predictions_train)
mse_test = mean_squared_error(y_test, predictions_test)
print("Mean Squared Error (Train):", mse_train)
print("Mean Squared Error (Test):", mse_test)

# Calculate coefficient of determination (R-squared) for training and testing sets
r2_train = r2_score(y_train, predictions_train)
r2_test = r2_score(y_test, predictions_test)
print("Coefficient of Determination (R-squared) (Train):", r2_train)
print("Coefficient of Determination (R-squared) (Test):", r2_test)

# 3D Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot the training data points
ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c='blue', marker='o', label='Train Data Points')

# Scatter plot the testing data points
ax.scatter(X_test[:, 0], X_test[:, 1], y_test, c='red', marker='x', label='Test Data Points')

# Calculate the regression plane
plane_x, plane_y = np.meshgrid(np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 100),
                               np.linspace(X_scaled[:, 1].min(), X_scaled[:, 1].max(), 100))
plane_z = coefficients[0] * plane_x + coefficients[1] * plane_y + intercept

# Plot the regression plane
ax.plot_surface(plane_x, plane_y, plane_z, color='green', alpha=0.3, label='Regression Plane')

# Set plot labels and title
ax.set_xlabel('acc')
ax.set_ylabel('consistency')
ax.set_zlabel('wpm')
ax.set_title('Linear Regression')

# Display the plot
plt.show()
