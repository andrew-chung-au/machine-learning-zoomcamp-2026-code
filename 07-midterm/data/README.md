# Dataset Attribution

## Phishing Websites

This directory contains the **Phishing Websites** dataset used in `reproduce_warning.py` to investigate a scikit-learn parallel-processing warning.

- **Source:** [UCI Machine Learning Repository — Phishing Websites](https://archive.ics.uci.edu/dataset/327/phishing)
- **Dataset DOI:** [10.24432/C51W2X](https://doi.org/10.24432/C51W2X)
- **Original authors:** Rami Mohammad and Lee McCluskey
- **Original publication:** *An assessment of features related to phishing websites using an automated technique* (2012)
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)

The dataset was originally published by the authors and is redistributed here under its stated licence. It contains integer-valued features for classifying websites as legitimate or phishing.

The reproducer loads `training-dataset.arff`, applies the preprocessing used in the homework, and runs a stratified cross-validation grid search with `RandomForestClassifier` and `GridSearchCV`.

For details of the dataset, its features, and the original research, consult the UCI source linked above.

