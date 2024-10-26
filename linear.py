import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
data = pd.read_csv('/home/student/Music/iris.csv')

# Print column names for verification
print("Column names:", data.columns)

# Clean column names (optional)
data.columns = data.columns.str.strip()  # Remove leading/trailing spaces

# Preprocess data
X = data.drop('variety', axis=1)  # Adjust to match your dataset's target variable
y = data['variety']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
