# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Read data from the CSV file
file_path = 'HN_housepricingdataset_2.csv'
df = pd.read_csv(file_path)

# Select numerical features and target variable
X = df[['District', 'Types of houses', 'Number of floors', 'Number of bedrooms', 'Area (m2)']]
y = df['Total price (million)']

# Standardize input data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Apply Polynomial Regression with degree 2
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train the Polynomial Regression model (Linear Regression)
model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)

# Predict Total price (million) of houses on the test set (Polynomial Regression)
y_pred_poly = model_poly.predict(X_test_poly)

# Evaluate Total price (million) with the Polynomial Regression model
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)

# Calculate R² on train and test sets
train_score = model_poly.score(X_train_poly, y_train)
test_score = model_poly.score(X_test_poly, y_test)

# Display resul
print(f"Polynomial Regression - Mean Squared Error (MSE): {mse_poly}")
print(f"Polynomial Regression - Root Mean Squared Error (RMSE): {rmse_poly}")
print(f"Polynomial Regression - R² (train): {train_score}")
print(f"Polynomial Regression - R² (test): {test_score}")

# Visualize predicted vs actual results (Polynomial Regression)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_poly, alpha=0.7, color='b', label='Predicted vs Actual (Polynomial)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2, label='Ideal Fit')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Polynomial Regression: Predicted vs Actual Prices')
plt.legend()
plt.show()

# Input data from the user
def predict_house_price():
    # Prompt for data input
    print("Enter the information to predict the Total price (million) of the house:")
    district = float(input("District: "))
    property_type = float(input("Types of houses: "))
    floors = float(input("Number of floors: "))
    bedrooms = float(input("Number of bedrooms: "))
    area = float(input("Area(m2): "))

    # Prepare input data with column names to avoid warnings
    input_data = pd.DataFrame([[district, property_type, floors, bedrooms, area]],
                              columns=['District', 'Types of houses', 'Number of floors', 'Number of bedrooms', 'Area (m2)'])

    # Standardize and transform input data
    input_data_scaled = scaler.transform(input_data)
    input_data_poly = poly.transform(input_data_scaled)

    # Predict Total price (million)
    predicted_price = model_poly.predict(input_data_poly)
    print(f"Predict the Total price (million) of the house: {predicted_price[0]:,.2f} million VND")

# Call the function for testing
predict_house_price()
