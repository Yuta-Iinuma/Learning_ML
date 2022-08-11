from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# データ読み込み
cancer = load_breast_cancer()

# トレーニングとテストに分割（#乱数固定）
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=1)

# ランダムフォレストモデル構築
# 学習パラメータ（以下直接使いそうなやつをpickup）
# n_estimators：バギングに使う決定技の数、デフォは10
# criterion：決定木の不純度算出のオプション、デフォはジニ不純度、その他にentropyなどが指定可
# max_depth：決定技のノードの深さ指定、おそらくデフォはNone、過学習にならない深さの調整が必要
model = RandomForestRegressor(random_state=0, max_depth=7)
model.fit(X_train, y_train)

# 評価
train_result = model.score(X_train, y_train)
test_result = result = model.score(X_test, y_test)

# print(train_result)
# print(test_result)


# 勾配ブースティングモデル構築
model = GradientBoostingRegressor(random_state=0)
model.fit(X_train, y_train)

# 評価
train_result = model.score(X_train, y_train)
test_result = result = model.score(X_test, y_test)

print(train_result)
print(test_result)
