from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris  # あやめのデータセットはsklearnに用意されている
import pandas as pd

# あやめのデータセットload
iris = load_iris()

# Pandasデータフレームにして見てみる
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

# 説明変数の設定
x = iris.data

# 目的変数の設定
y = iris.target

# 訓練データ，テストデータ分割（random_stateを指定しないと）
x_train, x_test, y_train, y_test = train_test_split(x, y)

# K-NN法とは
# まず、学習用データセットをラベルをつけてプロット
# k-meansみたいに空間上でラベル毎に分類する
# 入力用データをプロットしてk値の範囲内にある要素で一番多い要素のラベルを推定値として分類する

# k=1としてモデルを初期化し学習
k = 1
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(x_train, y_train)

# k-NNで識別
y_pred = knn_model.predict(x_test)

# 識別精度の算出
result = metrics.accuracy_score(y_test, y_pred)


result2 = knn_model.score(x_test, y_test)
# 識別結果から混同行列（Confusion Matrix）を作成
# 多クラスの混合行列ではどこで何が間違ったのかを把握するためのもの
# https://panda-clip.com/multiclass-confusion-matrix/

# 正しい値と予測値を使って作成
cm = confusion_matrix(y_test, y_pred)

# テーブル表示
print(cm)

# ヒートマップ表示
cm = ConfusionMatrixDisplay(cm, display_labels=iris.target_names)
cm.plot()
