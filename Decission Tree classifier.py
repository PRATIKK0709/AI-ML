import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree

# Load the dataset
df = pd.read_csv('./best-selling game consoles.csv')

# Cleaning up the data: Replace '0' in 'Discontinuation Year' with the maximum non-zero value
max_year = df[df['Discontinuation Year'] != 0]['Discontinuation Year'].max()
df['Discontinuation Year'].replace(0, max_year, inplace=True)

# Prepare the target and features for classification
# Encode the 'Type' column to have numerical values
le = LabelEncoder()
df['Type_encoded'] = le.fit_transform(df['Type'])

# Select features - we'll use 'Released Year', 'Discontinuation Year', and 'Units sold (million)'
X = df[['Released Year', 'Discontinuation Year', 'Units sold (million)']]
y = df['Type_encoded']  # Our target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train a Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the test data
accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Visualize the decision tree
plt.figure(figsize=(20, 10))
tree.plot_tree(model, feature_names=X.columns, class_names=le.classes_, filled=True)
plt.show()
