# Homework 5: Deploying Machine Learning Models

Below are my answers to the homework questions.

This homework deploys a frozen lead-scoring model trained on the 2026 course dataset. The model predicts whether a prospective lead will convert and sign up to the platform.

---

## 1. Environment check *(1 point)*

**Answer:**  
- My local `uv` version was `uv 0.11.20 (x86_64-unknown-linux-gnu)`.

---

## 2. Locked dependency *(1 point)*

**Answer:** 
- The Scikit-Learn version in the 2026 reference environment is `1.7.2`.

---

## 3. Load the model *(1 point)*

**Answer:**  
- The predicted probability of conversion is `0.533`.

---

## 4. Serve the model API *(1 point)*

**Answer:**   
- `conversion_probability`: `0.769799`
- `conversion`: `true`
- `conversion_probability` rounded to 3 decimal places: `0.770`

---

## 5. Container configuration *(1 point)*

**Answer:**  
- The Python base-image tag declared in the Dockerfile is `python:3.11.15-slim-bookworm`.

---

## 6. Run the container *(1 point)*

**Answer:**  
- The container returned a conversion probability of `0.769799`, which is `0.770` rounded to 3 decimal places.

---