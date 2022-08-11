# 線形分離の際は「LinearSVC」、非線形分離の場合は「SVC」
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd

# 乳がんデータセット
cancer = load_breast_cancer()

# Pandasデータフレームにして見てみる
breast_cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)

# SVMとは
# 教師あり学習で回帰・分類する機械学習アルゴリズム
# 線形分類する際の境界線の中で、各データから最も離れた境界線を分類線とする手法である。
# メリット：データの次元が大きくなっても分類精度が高い、最適化すべきパラメータが少ない、非線形分類もできる
# デメリット：学習データが増えると処理時間がかかる、基本的に2値分類に特化している、スケーリングが必要(SVMでは距離を測定するので、大きい範囲をとる特徴量に引きずられないようにする)


# 訓練データとテストデータに分ける
x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=0)

# SVMモデルの初期化と学習
svm_model = LinearSVC()
svm_model.fit(x_train, y_train)


# 訓練データの識別率
# y_pred = svm_model.predict(x_train)
# result = metrics.accuracy_score(y_train, y_pred)
# print(result)

# result = svm_model.score(x_train, y_train)
# print(result)

# テストデータの識別率
result = svm_model.score(x_test, y_test)
print(result)

y_pred = svm_model.predict(x_test)
result = metrics.accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
print(cm)
cm = ConfusionMatrixDisplay(cm, display_labels=cancer.target_names)
cm.plot()
