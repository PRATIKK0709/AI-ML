import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset from a CSV file
df = pd.read_csv(r'./best-selling game consoles.csv')
# Fill missing values for Discontinuation Year with the maximum year found in the dataset (assuming ongoing)
max_year = df['Discontinuation Year'].max()
df['Discontinuation Year'] = df['Discontinuation Year'].replace(0, max_year)

# Encode the 'Type' column to have numerical values
le = LabelEncoder()
df['Type_encoded'] = le.fit_transform(df['Type'])

# We'll use 'Released Year', 'Discontinuation Year', and 'Units sold (million)' as features
# and predict 'Type' of the console
X = df[['Released Year', 'Discontinuation Year', 'Units sold (million)']]
y = df['Type_encoded']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create an SVM model with the RBF kernel and fit it to the training data
rbf_model = SVC(kernel='rbf', gamma='auto')  # Using 'auto' for gamma to avoid warnings
rbf_model.fit(X_train, y_train)

# Predict the 'Type' on the test data
y_pred = rbf_model.predict(X_test)

# Calculate and print the accuracy score of the model on the test data
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# If you want to see what each encoded category represents, you can do this:
type_mappings = dict(zip(le.classes_, le.transform(le.classes_)))
print("Type mappings:", type_mappings)
