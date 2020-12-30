from sklearn.datasets import load_iris
from lightgbm import LGBMClassifier
from lgbp import plot_tree
import matplotlib.pyplot as plt

X, Y = load_iris(return_X_y=True)
clf = LGBMClassifier(n_estimators=5)
clf.fit(X, Y)
ax = plot_tree(clf, tree_index=1, figsize=(5, 5), dpi=400, show_info=None,
               precision=3, orientation='vertical')
plt.savefig('lgbm.png')
