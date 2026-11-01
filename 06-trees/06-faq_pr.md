**Question**
How do I read the output of `export_text` for a decision tree (what do <= 0.50 and > 0.50 mean, and which block is the left or right branch)?

**Answer**
**Top block = left branch = the feature is 0. Bottom block = right branch = the feature is 1.**

`export_text` prints the tree as text, so the two branches are stacked one above the other. If you draw the same tree with `plot_tree`, the top block becomes the left branch and the bottom block becomes the right branch.

`DictVectorizer` turns each category into a 0/1 column, such as `records=yes`. The tree can only ask "is this column 0 or 1?", and `export_text` writes that as `<= 0.50` (0) and `> 0.50` (1).

```python
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.feature_extraction import DictVectorizer

rows = [{'records': 'no'}] * 6 + [{'records': 'yes'}] * 4
y = [0, 0, 0, 0, 0, 1] + [1, 1, 1, 0]

dv = DictVectorizer(sparse=False)
X = dv.fit_transform(rows)
names = list(dv.get_feature_names_out())

dt = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)
print(export_text(dt, feature_names=names))
```

```text
|--- records=yes <= 0.50      <- top block (left branch): records=yes is 0
|   |--- class: 0
|--- records=yes >  0.50      <- bottom block (right branch): records=yes is 1
|   |--- class: 1
```

- Top (left): `records=yes` is 0, so the customer has no records, and the tree predicts class 0.
- Bottom (right): `records=yes` is 1, so the customer has records, and the tree predicts class 1.

**Watch for double negatives**

With a column like `records=no`, the top block means `records=no` is 0, which means the customer does have records. Turn each condition into plain words before reading the `class:` lines.

The `class: 0` and `class: 1` lines are the predictions. What they mean (for example, 1 = default) comes from how you encoded your target.