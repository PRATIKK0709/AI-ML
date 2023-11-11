import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('./best-selling game consoles.csv')
# Assuming you want to predict 'Type' based on other features
# Encode categorical text data to numerical format
label_encoder = LabelEncoder()
df['Type_encoded'] = label_encoder.fit_transform(df['Type'])
df['Company_encoded'] = label_encoder.fit_transform(df['Company'])

# Replace '0' in 'Discontinuation Year' with the maximum non-zero value
max_year = df[df['Discontinuation Year'] != 0]['Discontinuation Year'].max()
df['Discontinuation Year'].replace(0, max_year, inplace=True)

# Prepare the dataset for training
X = df[['Released Year', 'Discontinuation Year', 'Units sold (million)', 'Company_encoded']]  # Features
y = df['Type_encoded']  # Target variable

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the model
rf_classifier.fit(X_train, y_train)
# Make predictions
y_pred = rf_classifier.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
# Visualize one of the trees from the forest
tree = rf_classifier.estimators_[0]
# Set the size of the figure
plt.figure(figsize=(20, 10))
# Plot the tree
plot_tree(tree, feature_names=X.columns, class_names=label_encoder.classes_, filled=True)
# Show the plot
plt.show()