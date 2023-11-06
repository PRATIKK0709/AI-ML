import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('./best-selling game consoles.csv')

# For the purpose of linear regression, we will use "Released Year" as the independent variable
# and "Units sold (million)" as the dependent variable. We need to handle non-numeric values and missing values.

# Convert 'Units sold (million)' to numeric, forcing non-numeric values to NaN
data['Units sold (million)'] = pd.to_numeric(data['Units sold (million)'], errors='coerce')

# Drop rows with NaN values in 'Units sold (million)' or 'Released Year'
data = data.dropna(subset=['Units sold (million)', 'Released Year'])

# Now let's prepare the data for linear regression
X = data[['Released Year']]  # Independent variable
y = data['Units sold (million)']  # Dependent variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a linear regression object
regressor = LinearRegression()

# Train the model using the training sets
regressor.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regressor.predict(X_test)

# The coefficients
print('Coefficients: \n', regressor.coef_)
# The mean squared error
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))

# Return the regressor and test data for further use
(regressor, X_test, y_test, y_pred)
