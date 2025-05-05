import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
df = pd.read_csv('house_data.csv')

# 2. Preprocessing: Features and target variable
X = df[['square_feet', 'bedrooms', 'bathrooms']]  # Independent variables
y = df['price']  # Dependent variable (target)

# 3. Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize the model
model = LinearRegression()

# 5. Train the model
model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5  # Root Mean Squared Error
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')

# 8. Visualize the results (Optional)
plt.scatter(y_test, y_pred)
plt.xlabel('True Prices')
plt.ylabel('Predicted Prices')
plt.title('True vs Predicted Prices')
plt.show()

# 9. Model coefficients (Optional)
print(f'Model Coefficients: {model.coef_}')
