import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
inputData = pd.read_csv("./best-selling game consoles.csv")  # Replace with the path to your CSV file

# Analysis of Data
print(inputData.dtypes)
print(inputData.columns)
print("Data shape:", inputData.shape)
print(inputData.head())
print(inputData.describe())
print(inputData.info())

# Check for Null Values
print(inputData.isnull().sum())

# Replace 0 with the current year for 'Discontinuation Year' if it means the console is still in production
# Assuming that '0' is used in the dataset to denote an ongoing production, which is not accurate in terms of year value
current_year = pd.Timestamp.now().year
inputData['Discontinuation Year'] = inputData['Discontinuation Year'].replace(0, current_year)

# Convert 'Released Year' and 'Discontinuation Year' to numeric for correlation calculation
inputData['Released Year'] = pd.to_numeric(inputData['Released Year'], errors='coerce')
inputData['Discontinuation Year'] = pd.to_numeric(inputData['Discontinuation Year'], errors='coerce')

# Exclude non-numeric columns for correlation calculation
numeric_cols = inputData.select_dtypes(include=[np.number])
corr_matrix = numeric_cols.corr()

# Data Visualization
# Heatmap of correlation for the numerical data
plt.figure(figsize=(8, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Scatterplot - we'll use 'Released Year' vs 'Units sold (million)' as an example
# Since the actual columns for 'Age' and 'Glucose' were not provided, adjust this as needed.
# Ensure that 'Units sold (million)' is a numeric column
inputData['Units sold (million)'] = pd.to_numeric(inputData['Units sold (million)'], errors='coerce')
plt.figure(figsize=(8, 8))
sns.scatterplot(x="Released Year", y="Units sold (million)", data=inputData)
plt.title("Released Year vs Units Sold (million)")
plt.show()
