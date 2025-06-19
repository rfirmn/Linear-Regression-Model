# Ecommerce Customer Spending Prediction

This project presents a **Simple Linear Regression analysis** performed on an e-commerce customer dataset to explore and predict customer spending behavior.

## 📌 Project Overview

The main objective of this notebook is to:

- Analyze customer data from an e-commerce platform
- Build and interpret a Linear Regression model
- Identify key features influencing customer yearly spending

## 📂 Dataset Description

The dataset, `Ecommerce Customers`, contains customer behavior metrics such as:

- `Avg. Session Length`: Average session time in minutes
- `Time on App`: Time spent on the company app
- `Time on Website`: Time spent on the company website
- `Length of Membership`: Duration of customer membership (in years)
- `Yearly Amount Spent`: Total spending by the customer in a year (target variable)

## 📊 Steps Performed in the Notebook

### 1. Importing Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

Libraries such as `pandas`, `matplotlib`, and `seaborn` are used for data manipulation and visualization.

### 2. Data Loading & Exploration

- Data loaded using `pd.read_csv()`
- Initial exploration with `.info()`, `.describe()`, and `.head()`

### 3. Data Visualization

Visualizations include:

- Jointplot between `Time on App` and `Yearly Amount Spent`
- Pairplot for all features
- Linear regression plot (`lmplot`) between `Length of Membership` and `Yearly Amount Spent`

### 4. Feature Selection

```python
X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']
```

### 5. Splitting Data

Dataset is split into training and testing sets:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### 6. Model Building

A Linear Regression model is trained using `scikit-learn`:

```python
lm = LinearRegression()
lm.fit(X_train, y_train)
```

### 7. Coefficient Analysis

The model's learned coefficients are displayed to interpret the influence of each feature:

```python
pd.DataFrame(lm.coef_, X.columns, columns=['Coef'])
```

### 8. Predictions & Visualization

Predictions on the test set are made and compared visually:

```python
predictions = lm.predict(X_test)
sns.scatterplot(x=predictions, y=y_test)
```

### 9. Model Evaluation

Model performance is evaluated using:

- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)

### 10. Conclusion

From the coefficient analysis and visualizations, the `Length of Membership` was found to be the strongest predictor of yearly spending.

## 🚀 Final Notes

This notebook demonstrates a full pipeline of simple linear regression:

- From raw data to trained model
- With visualization and interpretability

It serves as a foundational example for machine learning beginners, especially in regression modeling.

---

Feel free to fork, experiment, and build on this project!

