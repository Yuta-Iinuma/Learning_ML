from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
# データ読み込み＆トレーニングとテストに分割
cancer = load_breast_cancer()
x_data = cancer.data
y_data = cancer.target

scores = {}
score_ar = {}
for depth_num in range(1, 21):
    for leaf_num in range(1, 21):
        tree_model = DecisionTreeClassifier(
            criterion='entropy', max_depth=depth_num, min_samples_leaf=leaf_num, random_state=0)
        scores = cross_val_score(tree_model, x_data, y_data, cv=50)
        score_ar[(depth_num, leaf_num)] = scores.mean()

score_series = pd.Series(score_ar)
print(score_series.max())
print(score_series.idxmax())
