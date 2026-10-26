**Question**
How do I fix `ValueError: Only sparse matrices with 32-bit integer indices are accepted. Got int64 indices` when fitting a model with DictVectorizer and scikit-learn?

**Answer**
### The Fix
Convert the sparse matrix index arrays to 32-bit integers immediately before passing them to `.fit()`:

```python
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

dv = DictVectorizer()
train_dict = df[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

# Force sparse matrix indices to 32-bit integers
X_train.indices = X_train.indices.astype(np.int32)
X_train.indptr = X_train.indptr.astype(np.int32)

model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)
```

### The Explanation
When using newer versions of NumPy (2.x) and scikit-learn in a fresh environment, `DictVectorizer` may output a sparse feature matrix whose internal pointer arrays use 64-bit integers (`int64`). However, certain scikit-learn estimators or solvers strictly enforce 32-bit integer indices (`int32`), causing model training to crash with a `ValueError`. Explicitly casting `.indices` and `.indptr` to `np.int32` resolves the type mismatch without altering your dataset or model architecture.