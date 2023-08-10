import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('m2.csv')

# Split the data into input (X) and output (y) variables
X = data[['acc', 'consistency']]
y = data['wpm']

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions using the trained model
predictions = model.predict(X)

# Print the coefficient values
coefficients = model.coef_
intercept = model.intercept_
for i, name in enumerate(X.columns):
    print(f"Coefficient for {name}: {coefficients[i]}")
print(f"Intercept: {intercept}")

# Calculate mean squared error
mse = mean_squared_error(y, predictions)
print("Mean Squared Error:", mse)

# Calculate coefficient of determination (R-squared)
r2 = r2_score(y, predictions)
print("Coefficient of Determination (R-squared):", r2)

# Plot the predicted values
plt.scatter(X['acc'], predictions, color='red', label='Predicted')

# Calculate the regression line
line_x = np.linspace(X['acc'].min(), X['acc'].max(), 100)
line_y = coefficients[0] * line_x + coefficients[1] * X['consistency'].mean() + intercept

# Plot the regression line
plt.plot(line_x, line_y, color='green', linewidth=2, label='Regression Line')

# Set plot labels and title
plt.xlabel('acc & consistency')
plt.ylabel('wpm')
plt.title('Linear Regression')

# Add a legend
plt.legend()

# Display the plot
plt.show()

