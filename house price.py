import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.ticker import FuncFormatter

# Sample house price data
data = {
    "area": [650, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600,
             1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600],
    "bedrooms": [1, 2, 2, 2, 2, 3, 3, 3, 3, 3,
                 3, 3, 4, 4, 4, 4, 4, 4, 4, 4],
    "floors": [1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
               3, 3, 3, 3, 3, 3, 3, 3, 4, 4],
    "age": [20, 15, 12, 10, 8, 7, 6, 5, 4, 3,
            5, 4, 6, 5, 4, 3, 2, 5, 4, 3],
    "price": [1800000, 2400000, 2800000, 3400000, 3900000,
              4500000, 5000000, 5400000, 5800000, 6300000,
              6700000, 7200000, 7600000, 8200000, 8800000,
              9400000, 10000000, 10300000, 11000000, 11700000]
}

data = pd.DataFrame(data)

# Features and target
X = data[["area", "bedrooms", "floors", "age"]]
y = data["price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Test prediction
y_pred = model.predict(X_test)

# Model evaluation
print("\n===== HOUSE PRICE PREDICTION =====")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):,.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

# User input
print("\nEnter house details:")

area = float(input("Enter area (sq ft): "))
bedrooms = int(input("Enter number of bedrooms: "))
floors = int(input("Enter number of floors: "))
age = int(input("Enter age of the house: "))

# Create input
input_data = pd.DataFrame(
    [[area, bedrooms, floors, age]],
    columns=["area", "bedrooms", "floors", "age"]
)

# Predict price
predicted_price = model.predict(input_data)

print("\n================================")
print(f"Predicted House Price: ₹{predicted_price[0]:,.2f}")
print("================================")

# Graph
plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7,
    label="Predicted Prices"
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--",
    label="Prediction Line"
)

plt.xlabel("Actual Prices (₹)")
plt.ylabel("Predicted Prices (₹)")
plt.title("Actual vs Predicted House Prices")

formatter = FuncFormatter(lambda x, pos: f"₹{int(x):,}")
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_formatter(formatter)

plt.legend()
plt.grid(True)
plt.show()
