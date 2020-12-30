# LightGBM_Plot
An extension to lightgbm plot module to make interpretable and useful lightgbm tree models

[![License](https://img.shields.io/github/license/KiLJ4EdeN/LightGBM_Plot)](https://img.shields.io/github/license/KiLJ4EdeN/LightGBM_Plot) [![Code size](https://img.shields.io/github/languages/code-size/KiLJ4EdeN/LightGBM_Plot)](https://img.shields.io/github/languages/code-size/KiLJ4EdeN/LightGBM_Plot) [![Repo size](https://img.shields.io/github/repo-size/KiLJ4EdeN/LightGBM_Plot)](https://img.shields.io/github/repo-size/KiLJ4EdeN/LightGBM_Plot) [![Open Issues](https://img.shields.io/github/issues/KiLJ4EdeN/LightGBM_Plot)](https://img.shields.io/github/issues/KiLJ4EdeN/LightGBM_Plot)
![Closed Issues](https://img.shields.io/github/issues-closed/KiLJ4EdeN/LightGBM_Plot)


## Demo output:
<img src="https://github.com/KiLJ4EdeN/LightGBM_Plot/blob/main/output.png" width="50%" height="50%" />


## How To Use:
```bash
git clone https://github.com/KiLJ4EdeN/LightGBM_Plot
cd LightGBM_Plot
pip install -r requirements.txt
python example.py
```

## Add the Plots Easily to Existing Problems:
```python
from sklearn.datasets import load_iris
from lightgbm import LGBMClassifier
from lgbp import plot_tree
import matplotlib.pyplot as plt

X, Y = load_iris(return_X_y=True)
clf = LGBMClassifier(n_estimators=5)
clf.fit(X, Y)
ax = plot_tree(clf, tree_index=1, figsize=(5, 5), dpi=400, show_info=None,
               precision=3, orientation='vertical')
plt.savefig('output.png')
```
