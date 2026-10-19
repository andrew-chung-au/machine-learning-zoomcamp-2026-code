**Question**
How do I avoid getting a `KeyError: 'converted'` when re-running the cell that deletes the target variable?

**Answer**
This error happens because Jupyter notebooks execute cells out of order or multiple times. Once you delete the target variable from your training dataframe using `del df_train['converted']`, the column is permanently removed from `df_train`. Running that same cell a second time triggers a `KeyError` because the column no longer exists.

**The Fix:**
You can use pandas `.drop()` with `errors='ignore'` so that re-running the cell won't crash if the column has already been removed:

```python
# Safe way to remove the target variable on multiple runs
df_train = df_train.drop(columns=['converted'], errors='ignore')
df_val = df_val.drop(columns=['converted'], errors='ignore')
df_test = df_test.drop(columns=['converted'], errors='ignore')