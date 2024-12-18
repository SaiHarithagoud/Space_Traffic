import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle

data = pd.read_csv(r"C:\Users\saiha\OneDrive\Desktop\space_traffic_app\space_traffic.csv")
X = data[['Location', 'Object_Type', 'Peak_Time']]
y = data['Traffic_Density']

preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(drop='first'), ['Location', 'Object_Type', 'Peak_Time'])],
    remainder='passthrough'
)

X_transformed = preprocessor.fit_transform(X).toarray()
y = np.array(y).reshape(-1, 1)
X_transformed = np.c_[np.ones((X_transformed.shape[0], 1)), X_transformed]

X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

weights = np.zeros((X_train.shape[1], 1))
learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):
    predictions = np.dot(X_train, weights)
    errors = predictions - y_train
    gradient = (1 / len(X_train)) * np.dot(X_train.T, errors)
    weights -= learning_rate * gradient
    if (epoch + 1) % 100 == 0:
        mse = mean_squared_error(y_train, predictions)
        print(f"Epoch {epoch + 1}/{epochs}, MSE: {mse:.2f}")

y_test_pred = np.dot(X_test, weights)
test_mse = mean_squared_error(y_test, y_test_pred)
print(f"Test Mean Squared Error: {test_mse:.2f}")

model_filename = 'trained_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump({'weights': weights, 'preprocessor': preprocessor}, f)

example_data = pd.DataFrame({
    'Location': ['Orbit LEO'],
    'Object_Type': ['Satellite'],
    'Peak_Time': ['15:00']
})
example_transformed = preprocessor.transform(example_data).toarray()
example_transformed = np.c_[np.ones((example_transformed.shape[0], 1)), example_transformed]
example_prediction = np.dot(example_transformed, weights)
print(f"Predicted Traffic Density for example: {example_prediction[0][0]:.2f}")