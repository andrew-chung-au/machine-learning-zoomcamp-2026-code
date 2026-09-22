**Question**
Why are there still missing values after I use the `fillna()` method?

**Answer**
By default, Pandas operations like `.fillna()` do not modify the original DataFrame; they return a modified copy of it. If you do not assign this copy back to a variable, your changes are immediately lost.

*   **The Fix:** Assign the result back to the specific column, like this: `df['score'] = df['score'].fillna(fill_value)`.
*   **Alternative Fix:** You can instruct Pandas to modify the existing DataFrame directly by using the `inplace` argument: `df['score'].fillna(fill_value, inplace=True)`.

Here is a runnable example demonstrating how to properly apply this when cleaning datasets (such as train, validation, and test sets)[cite: 1]:

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({"score": [100, np.nan, 150]})
fill_value = 0 

# INCORRECT: The change is lost
df["score"].fillna(fill_value)
print(df["score"].isnull().sum()) # Output is 1 (still missing)

# CORRECT: Reassign the column
df["score"] = df["score"].fillna(fill_value)
print(df["score"].isnull().sum()) # Output is 0 (fixed)