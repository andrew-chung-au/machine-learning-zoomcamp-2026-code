**Question**
In lesson 2.13 (Regularization), adding a tiny bit of noise (5.00000001) to the duplicated column is supposed to make the matrix invertible and produce huge weights. Why do I get LinAlgError: Singular matrix instead?

**Answer**
You didn't make a mistake. This is a floating-point precision difference between your environment and the one used in the video.

In the lesson, `5.00000001` makes the duplicated columns *slightly* different, so the instructor's NumPy finds an inverse with huge values (around 10^14). But the difference is so small (1e-8) that after computing `X.T @ X`, it can get lost in rounding. Depending on your NumPy version and linear algebra backend (OpenBLAS, MKL, Apple Accelerate), the Gram matrix may be treated as exactly singular, so `np.linalg.inv` raises `LinAlgError` instead of returning huge numbers.

Notice that the smaller 3×3 example later in the same lesson (with `1.0000001`) still inverts and gives values around 5 million. The noise there is larger, so it isn't lost to rounding.

**To see the behaviour shown in the video**, use a slightly larger noise value:

```python
X = np.array([
    [4, 4, 4],
    [3, 5, 5],
    [5, 1, 1],
    [5, 4, 4],
    [7, 5, 5],
    [4, 5, 5.0000001],   # one more decimal place of noise
])
XTX = X.T @ X
np.linalg.inv(XTX)       # huge values, around 10^13
```

**To fix it**, apply regularization, which is the point of the lesson. Add a small number to the diagonal before inverting:

```python
XTX = XTX + 0.01 * np.eye(3)
np.linalg.inv(XTX)       # works; values are now under control
```

This works with `5.00000001` too. In your own training function, use `np.eye(XTX.shape[0])` instead of `np.eye(3)` so it matches the number of features, as in `train_linear_regression_reg`.