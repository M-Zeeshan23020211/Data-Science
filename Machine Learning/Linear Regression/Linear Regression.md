# Linear Regression in Machine Learning

## What is Regression?

Regression is a type of **supervised machine learning** used to predict a **continuous numerical value** from one or more input features. The model learns the relationship between inputs (independent variables) and an output (dependent variable) from labeled training data.

**Examples:**
- Predicting house prices from size, location, and number of bedrooms
- Forecasting sales revenue from advertising spend
- Estimating a patient's blood pressure from age and weight

> **Regression vs. Classification:** Regression predicts a *number* (e.g., 245,000). Classification predicts a *category* (e.g., "spam" or "not spam").

---

## How It Works

1. **Collect data** with input features (`X`) and known target values (`y`).
2. **Choose a model** that defines the form of the relationship.
3. **Train the model** by adjusting its parameters to minimize a *loss function* (the gap between predictions and actual values).
4. **Evaluate** on unseen test data.
5. **Predict** values for new data.

For simple linear regression, the model is:

```
y = β₀ + β₁x + ε
```

- `y`: predicted value
- `x`: input feature
- `β₀`: intercept
- `β₁`: slope (coefficient)
- `ε`: error term

For multiple features:

```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

---

## Common Types of Regression

| Model | Description | Best For |
|-------|-------------|----------|
| **Linear Regression** | Fits a straight line (or hyperplane) to the data | Simple, interpretable relationships |
| **Polynomial Regression** | Fits a curved line using polynomial terms | Non-linear trends |
| **Ridge Regression (L2)** | Linear regression with a penalty on large coefficients | Reducing overfitting, correlated features |
| **Lasso Regression (L1)** | Penalizes coefficients and can shrink some to zero | Feature selection |
| **Elastic Net** | Combines L1 and L2 penalties | Many correlated features |
| **Decision Tree Regression** | Splits data into regions and predicts the average of each | Non-linear data, easy to visualize |
| **Random Forest Regression** | Averages many decision trees | Strong accuracy, robust to noise |
| **Gradient Boosting (XGBoost, LightGBM)** | Builds trees sequentially to correct earlier errors | High-performance tabular predictions |
| **Support Vector Regression (SVR)** | Fits within a margin of tolerance around the data | Smaller datasets, non-linear patterns |

---

## Loss Functions

The loss function measures how wrong the model's predictions are. Training aims to minimize it.

- **Mean Squared Error (MSE):** average of squared errors. Penalizes large errors heavily.
- **Mean Absolute Error (MAE):** average of absolute errors. More robust to outliers.
- **Huber Loss:** blends MSE and MAE for outlier robustness.

---

## Evaluation Metrics

| Metric | Meaning |
|--------|---------|
| **MAE** | Average absolute difference between predicted and actual values |
| **MSE** | Average squared difference |
| **RMSE** | Square root of MSE (same units as the target) |
| **R² (R-squared)** | Proportion of variance explained by the model (1.0 = perfect, 0 = no better than predicting the mean) |

---

## Key Assumptions of Linear Regression

- **Linearity:** the relationship between features and target is linear
- **Independence:** observations are independent of each other
- **Homoscedasticity:** the variance of errors is constant
- **Normality of errors:** residuals are approximately normally distributed
- **No multicollinearity:** features are not highly correlated with each other

---

## Example: Linear Regression in Python

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample data: house size (sq ft) vs. price
X = np.array([[800], [1000], [1200], [1500], [1800], [2000], [2400]])
y = np.array([150000, 180000, 210000, 260000, 300000, 340000, 400000])

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²:", r2_score(y_test, y_pred))

# Predict a new value
print("Predicted price for 1,600 sq ft:", model.predict([[1600]])[0])
```

---

## Common Challenges

- **Overfitting:** the model memorizes training data and performs poorly on new data. *Fix:* regularization (Ridge/Lasso), cross-validation, more data.
- **Underfitting:** the model is too simple to capture the pattern. *Fix:* more complex model, better features.
- **Outliers:** extreme values can skew results. *Fix:* robust loss functions, outlier removal.
- **Multicollinearity:** correlated features make coefficients unstable. *Fix:* remove features, use Ridge regression, or apply PCA.
- **Feature scaling:** some models (SVR, Ridge, Lasso) perform better when features are standardized.

---

## Pros and Cons

**Advantages**
- Simple and fast to train (especially linear models)
- Highly interpretable
- Works well as a baseline model

**Limitations**
- Linear models can't capture complex non-linear patterns
- Sensitive to outliers
- Performance depends heavily on feature quality

---

## When to Use Regression

Use regression when your target variable is **continuous** and you want to understand or predict how it changes with other variables.

---

## Further Reading

- [Scikit-learn: Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- [Scikit-learn: Regression Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
