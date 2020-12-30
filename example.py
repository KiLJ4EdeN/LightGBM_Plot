from sklearn.datasets import load_breast_cancer
from lightgbm import LGBMClassifier
from lgbp import plot_tree
import matplotlib.pyplot as plt

X, Y = load_breast_cancer(return_X_y=True, as_frame=True)
clf = LGBMClassifier(n_estimators=5)
clf.fit(X, Y)
ax = plot_tree(clf, tree_index=0, figsize=(5, 5), dpi=400, show_info=None,
               precision=3, orientation='vertical')
plt.savefig('lgbm.png')
