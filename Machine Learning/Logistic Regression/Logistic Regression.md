# All About Logistic Regression Read and Practic

## What is Logistic Regression?

Logistic regression is a **supervised machine learning** algorithm used for **classification**. Despite the name, it predicts a **category** (such as yes/no or pass/fail), not a continuous number.

It outputs a **probability between 0 and 1**, then turns that probability into a class label.

**Examples:**
- Will a student pass or fail based on hours studied?
- Is an email spam or not spam?
- Will a customer churn or stay?

> **Linear vs. Logistic Regression:** Linear regression predicts a number (e.g., a house price). Logistic regression predicts the probability of a class (e.g., 0.82 chance of passing).

---

## How It Works

### 1. Start with a linear equation

```
z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

### 2. Squash it with the sigmoid function

The sigmoid function converts any number `z` into a value between 0 and 1:

```
p = 1 / (1 + e^(-z))
```

- If `z` is large and positive, `p` is close to **1**
- If `z` is large and negative, `p` is close to **0**
- If `z = 0`, `p = 0.5`

### 3. Apply a decision threshold

```
if p >= 0.5  →  predict class 1
if p <  0.5  →  predict class 0
```

The 0.5 threshold is the default but can be adjusted depending on the problem.

---

## How the Model Learns

The model finds the coefficients (`β`) that minimize **log loss** (also called binary cross-entropy):

```
Loss = -[ y·log(p) + (1 - y)·log(1 - p) ]
```

Log loss heavily penalizes confident wrong predictions. This is usually minimized with an optimization method such as gradient descent.

---

## Simple Example: Will a Student Pass?

**Goal:** Predict whether a student passes (1) or fails (0) based on hours studied.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Data: hours studied and result (0 = fail, 1 = pass)
hours = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4,
                  4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]).reshape(-1, 1)
passed = np.array([0, 0, 0, 0, 0, 0, 1, 0,
                   1, 0, 1, 1, 1, 1, 1, 1])

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    hours, passed, test_size=0.25, random_state=42, stratify=passed
)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict classes and probabilities
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # probability of passing

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Predict for a new student who studied 4.5 hours
new_student = np.array([[4.5]])
print("Predicted class:", model.predict(new_student)[0])
print("Probability of passing:", model.predict_proba(new_student)[0][1])
```

**What the output means:**
- `predict()` returns the final class (0 or 1).
- `predict_proba()` returns the probability of each class.
- Results will vary slightly because the dataset is tiny. This is just for learning.

---

## Visualizing the Sigmoid Curve

```python
import matplotlib.pyplot as plt

x_range = np.linspace(0, 9, 200).reshape(-1, 1)
probs = model.predict_proba(x_range)[:, 1]

plt.scatter(hours, passed, color="black", label="Data")
plt.plot(x_range, probs, color="blue", label="Predicted probability")
plt.axhline(0.5, color="red", linestyle="--", label="Threshold (0.5)")
plt.xlabel("Hours Studied")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression")
plt.legend()
plt.show()
```

You'll see an **S-shaped curve** rising from 0 to 1 as hours studied increases.

---

## Evaluation Metrics

| Metric | Meaning |
|--------|---------|
| **Accuracy** | Percentage of correct predictions |
| **Precision** | Of all predicted positives, how many were actually positive |
| **Recall** | Of all actual positives, how many were found |
| **F1-Score** | Balance between precision and recall |
| **ROC-AUC** | How well the model separates the two classes across all thresholds |

### Confusion Matrix

```
                 Predicted 0   Predicted 1
Actual 0            TN             FP
Actual 1            FN             TP
```

- **TP** (True Positive): predicted pass, actually passed
- **TN** (True Negative): predicted fail, actually failed
- **FP** (False Positive): predicted pass, actually failed
- **FN** (False Negative): predicted fail, actually passed

> **Tip:** Accuracy can be misleading when classes are imbalanced. Check precision, recall, and F1 as well.

---

## Types of Logistic Regression

| Type | Description | Example |
|------|-------------|---------|
| **Binary** | Two classes | Spam vs. not spam |
| **Multinomial** | Three or more unordered classes | Cat, dog, or bird |
| **Ordinal** | Three or more ordered classes | Low, medium, high |

---

## Key Assumptions

- The target variable is **categorical** (binary for the basic version)
- Observations are **independent**
- **Little to no multicollinearity** between features
- A **linear relationship** between features and the *log-odds* of the outcome
- A reasonably **large sample size**

---

## Pros and Cons

**Advantages**
- Simple, fast, and easy to implement
- Highly interpretable (coefficients show each feature's influence)
- Outputs probabilities, not just labels
- Strong baseline for classification problems

**Limitations**
- Assumes a linear decision boundary
- Can struggle with complex, non-linear patterns
- Sensitive to outliers and highly correlated features
- Needs feature scaling and regularization for best results on larger problems

---

## Tips for Better Results

- **Scale your features** (e.g., with `StandardScaler`) when using multiple features
- Use **regularization** (`C` parameter in scikit-learn; smaller `C` means stronger regularization)
- Handle **class imbalance** with `class_weight="balanced"`
- Tune the **decision threshold** if false positives and false negatives have different costs

---

## Further Reading

- [Scikit-learn: Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Scikit-learn: Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
