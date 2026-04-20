# 1. Convert text/categories into numbers (One-Hot Encoding)
df_encoded = pd.get_dummies(df, columns=['Location', 'Neighborhood', 'Condition'], drop_first=True)

# 2. Update X to include everything except the target 'Price' and 'Id'
X = df_encoded.drop(['Price', 'Id'], axis=1) 
y = df_encoded['Price'] 
# --- Updated Plotting Section ---
plt.figure(figsize=(10, 6)) # Makes the graph bigger

# Your actual data points
plt.scatter(y_test, y_pred, alpha=0.5, color='royalblue', label='Model Predictions')

# --- This code creates the straight line ---
# It finds the lowest and highest prices to draw a perfect diagonal
max_val = max(max(y_test), max(y_pred))
min_val = min(min(y_test), min(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Prediction (Goal)')
# ------------------------------------------

plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.legend() # Shows what the dots and line mean
plt.grid(True, linestyle=':', alpha=0.6) # Adds a subtle grid
plt.show()

print("\nProject Completed Successfully!")
