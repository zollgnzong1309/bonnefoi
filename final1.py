# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load normalized dataset
file_path = 'data_14_normalized.csv'  # Normalized dataset
normalized_df = pd.read_csv(file_path)

# Display column names to verify structure
print(normalized_df.columns)

# Define features and target
X = normalized_df[['District', 'Floors', 'Rooms', 'Area']]
y = normalized_df['Total Price (million)']

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train a Polynomial Regression model
model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)

# Predict on the test set
y_pred_poly = model_poly.predict(X_test_poly)

# Evaluate the model
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly_test = r2_score(y_test, y_pred_poly)
r2_poly_train = model_poly.score(X_train_poly, y_train)

# Display results
print(f"Polynomial Regression - Mean Squared Error (MSE): {mse_poly}")
print(f"Polynomial Regression - R² (train): {r2_poly_train}")
print(f"Polynomial Regression - R² (test): {r2_poly_test}")

# Visualization: Predicted vs Actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_poly, alpha=0.7, color='b', label='Predicted vs Actual (Polynomial)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2, label='Ideal Fit')
plt.xlabel('Actual Price (million)')
plt.ylabel('Predicted Price (million)')
plt.title('Polynomial Regression: Predicted vs Actual Prices')
plt.legend()
plt.show()

# User-defined prediction function
def predict_house_price():
    print("Enter the number of samples to predict:")
    n = int(input("Number of samples: "))
    results = []

    for i in range(n):
        print(f"Enter information for sample {i + 1}:")
        district = float(input("District (normalized value): "))
        floors = float(input("Number of floors (normalized value): "))
        rooms = float(input("Number of rooms (normalized value): "))
        area = float(input("Area (m2, normalized value): "))

        # Prepare input data
        input_data = np.array([[district, floors, rooms, area]])
        input_data_poly = poly.transform(input_data)

        # Predict price
        predicted_price = model_poly.predict(input_data_poly)
        results.append([district, floors, rooms, area, predicted_price[0]])

    # Convert results to DataFrame
    result_df = pd.DataFrame(results, columns=['District', 'Floors', 'Rooms', 'Area', 'Predicted Price (million)'])
    print("\nPrediction Results:")
    print(result_df)

    # Save results to CSV
    result_df.to_csv("predicted_results.csv", index=False)
    print("Results have been saved to file predicted_results.csv")

# Call prediction function
predict_house_price()
