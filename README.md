Code Breakdown and Explanation
1. Import Necessary Libraries
python
Copy code
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
What It Does: Loads essential libraries.
pandas for handling data.
numpy for numerical operations.
sklearn for machine learning tasks like splitting data, modeling, and evaluation.
matplotlib and seaborn for visualization.
2. Load the Dataset
python
Copy code
df = pd.read_csv("movie_revenue_dataset.csv")
What It Does: Reads the dataset from a CSV file into a Pandas DataFrame.
Expected Output: A table containing movie data with columns like genre, budget, cast_popularity, and revenue.
3. One-Hot Encoding
python
Copy code
df = pd.get_dummies(df, columns=['genre'], drop_first=True)
What It Does: Converts the categorical column genre into numeric columns using one-hot encoding (e.g., if there are 4 genres, it creates 3 binary columns to represent them).
Why? Machine learning models work better with numeric data.
4. Splitting Features and Target
python
Copy code
X = df.drop('revenue', axis=1)  # Features
y = df['revenue']  # Target variable
What It Does:
X contains all columns except revenue (the target variable).
y contains the revenue column.
5. Train-Test Split
python
Copy code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
What It Does: Splits the data into training (80%) and testing (20%) sets.
Why? This ensures the model is trained on one subset and tested on another for unbiased evaluation.
6. Build the Model
python
Copy code
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
What It Does:
Initializes a Random Forest Regressor with 100 trees.
Fits (trains) the model using the training data (X_train and y_train).
7. Make Predictions
python
Copy code
y_pred = model.predict(X_test)
What It Does: Predicts movie revenues for the test data using the trained model.
8. Evaluate the Model
python
Copy code
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
What It Does:
MSE (Mean Squared Error): Measures the average squared difference between actual and predicted values (lower is better).
R² Score: Measures how well the model explains variance in the data (closer to 1 is better).
Example Output:

yaml
Copy code
Mean Squared Error: 1234567.89
R-squared Score: 0.85
9. Visualize Results
python
Copy code
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Actual vs Predicted Movie Revenue")
plt.show()
What It Does: Creates a scatter plot of actual vs. predicted values.
Expected Plot:
Ideal Case: Points aligned along the diagonal.
Deviation: Indicates prediction errors.
10. Feature Importance
python
Copy code
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance)
plt.title("Feature Importance")
plt.show()
What It Does:
model.feature_importances_ shows how much each feature contributes to predictions.
Creates a bar plot ranking feature importance.
Residual Graph
Residuals = Difference between actual and predicted values.

python
Copy code
residuals = y_test - y_pred

# Residual Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=residuals, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Actual Revenue")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()
What It Shows:
Points close to the horizontal line (y=0) indicate better predictions.
Larger deviations show areas where the model performs poorly.
