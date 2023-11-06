import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

data = pd.read_csv(r'./best-selling game consoles.csv')
data['Is Discontinued'] = data['Discontinuation Year'].apply(lambda x: 0 if x == 0 else 1)

data.drop(['Discontinuation Year', 'Remarks', 'Console Name'], axis=1, inplace=True)
X = data.drop('Is Discontinued', axis=1)
y = data['Is Discontinued']

categorical_features = ['Type', 'Company']
numeric_features = ['Released Year', 'Units sold (million)']  # Assuming these are the only numeric columns
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Updated parameter name
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'  # Numeric features are passed through
)

nb_classifier = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', CategoricalNB())
])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_classifier.fit(X_train, y_train)

y_pred = nb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
