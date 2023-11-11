import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (assuming it is stored in 'console_sales.csv')
df = pd.read_csv(r'./best-selling game consoles.csv')

# Preprocessing: Encode categorical data
label_encoder = LabelEncoder()
df['Company_encoded'] = label_encoder.fit_transform(df['Company'])
df['Type_encoded'] = label_encoder.fit_transform(df['Type'])

# Replace 'Discontinuation Year' placeholder zeros with the maximum non-zero value for ongoing sales
df['Discontinuation Year'].replace(0, df['Discontinuation Year'].max(), inplace=True)

# Select features and target for the model
X = df[['Released Year', 'Discontinuation Year', 'Units sold (million)', 'Company_encoded']]
y = df['Type_encoded']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning using GridSearchCV
param_grid = {'n_neighbors': [1, 3, 5, 7, 10]}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_n_neighbors = grid_search.best_params_['n_neighbors']

# Create and train the KNN classifier with the best n_neighbors value
knn = KNeighborsClassifier(n_neighbors=best_n_neighbors)
knn.fit(X_train, y_train)

# Predict on the test data
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

# Generate a classification report
classification_rep = classification_report(y_test, y_pred)
print(classification_rep)

# Show the plot
plt.show()
