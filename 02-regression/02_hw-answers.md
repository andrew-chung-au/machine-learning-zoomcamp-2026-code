# Homework 2: Regression

Below are my answers to the homework questions.

This homework creates a linear regression model to predict car fuel efficiency using the `fuel_efficiency_mpg` target variable.

---

## 1. Column with missing values *(1 point)*

**Answer:**  
- `horsepower`

---

## 2. Median horsepower *(1 point)*

**Answer:**  
- `254`

---

## 3. Missing-value imputation *(1 point)*

**Answer:**  
-  `With mean`

---

## 4. Regularized linear regression *(1 point)*

**Answer:**  
- `0`

---

## 5. Effect of random seed *(1 point)*

**Answer:**  
- `[ADD ANSWER: 0.006 / 0.016 / 0.029 / 0.036]`

I repeated the train/validation/test split using seeds from `0` to `9`. For each split, I filled missing values with `0`, trained an unregularized linear regression model, and calculated validation RMSE.

---

## 6. Final model evaluation *(1 point)*

**Answer:**  
- `2.236`

---