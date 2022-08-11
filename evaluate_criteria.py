from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

iris = load_iris()  # あやめのデータセット
x = iris.data  # 特徴量（入力，説明変数）
y = iris.target  # 目的変数（出力，クラス，カテゴリ）
# random_stateを指定しないとnp.randomを用いた乱数セットになる
X_train, X_test, y_train, y_test = train_test_split(x, y)
knn_model = KNeighborsClassifier(n_neighbors=1)  # モデル初期化
knn_model.fit(X_train, y_train)  # 学習
y_pred = knn_model.predict(X_test)  # 認識
m = confusion_matrix(y_test, y_pred)  # 混同行列作成


# 正解率
# 全てのサンプルのうち、正解したサンプルの割合
accuracy = (m[0, 0]+m[1, 1]+m[2, 2]) / m.sum()
print('正解率')
print(accuracy)

# 再現率（各カテゴリ（クラス）別）
# 予測値と実数値の乖離を評価
#　実際に陽性のサンプルのうち、正解したサンプルの割合
# TP/TP+FN
rs = recall_score(y_test, y_pred, average=None)
print('再現率')
print(rs)

# マクロ平均とマイクロ平均
# 多クラス分類する場合
# マクロ平均は、各クラスの各数値（再現率や適合率）の平均をとる
# マイクロ平均は、各クラスの各数値の算出の際の要素となる数値（適合率ならTP,FP　再現率ならTP,FNなど）単位で平均を取る

# 再現率（平均）
rs = recall_score(y_test, y_pred, average='macro')
print('再現率の平均')
print(rs)

# 適合率（各カテゴリ（クラス）別）
# 陽性と予測されたサンプルのうち、正解のサンプルの割合
# TP/TP+FP
ps = precision_score(y_test, y_pred, average=None)
print('適合率')
print(ps)

# 適合率（平均）
ps = precision_score(y_test, y_pred, average='macro')
print('適合率の平均')
print(ps)

# F値（各カテゴリ（クラス）別）
# 適合率と再現率の調和平均
fs = f1_score(y_test, y_pred, average=None)
print('F値')
print(fs)

# F値（平均）
fs = f1_score(y_test, y_pred, average='macro')
print('F値の平均')
print(fs)
