import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

diabetes = datasets.load_diabetes()
X = diabetes.data[:, np.newaxis, 2]
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


model = linear_model.LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print(y_pred)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")


plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='black', label="Actual Data")
plt.plot(X_test, y_pred, color='blue', linewidth=1, label='Predicted Line')
plt.title("Actual vs Predicted Values")
plt.xlabel('BMI')
plt.ylabel('Disease Progression')
plt.legend()
plt.grid(True)
plt.show()
