from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from pandas.core.dtypes.cast import soft_convert_objects
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

#乳がんデータセットを利用
cancer = load_breast_cancer()

#トレーニングとテストに分割
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify = cancer.target, random_state=0) 

scores = {}#スコア蓄積用の辞書型変数

#gammaとCの組み合わせ49通りで学習＆識別
for gamma in np.logspace(-3, 3, num=7):
    for C in np.logspace(-3, 3, num=7):
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        scores[(gamma, C)] = svm.score(X_test, y_test)

#最大値などのできるようにSeries型にする
scores = pd.Series(scores)

#ベストスコアとその時のハイパーパラメータ（gamma, C）
print(scores.max())
print(scores.idxmax())

#結果を表で表示
result= scores.unstack()
print(result)

#結果をヒートマップで表示
sns.heatmap(result, cmap='RdYlGn')
plt.xlabel("C")
plt.ylabel("gamma")
plt.tight_layout()