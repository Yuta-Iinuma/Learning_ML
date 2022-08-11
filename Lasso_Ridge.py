from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import requests
import zipfile
import io
import pandas as pd
import numpy as np

# 自動車価格データを取得
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
res = requests.get(url).content

# 取得したデータをDataFrameオブジェクトとして読み込み
auto = pd.read_csv(io.StringIO(res.decode('utf-8')), header=None)

# データの列にラベルを設定
auto.columns = ['symboling', 'normalized_losses', 'make', 'fuel_type', 'aspiration', 'num_of_doors',
                'body_style', 'drive_wheels', 'engine_location', 'wheel_base', 'length', 'width', 'height',
                'curb_weight', 'engine_type', 'num_of_cylinders', 'engine_size', 'fuel_system', 'bore',
                'stroke', 'compression_ratio', 'horsepower', 'peak_rpm', 'city_mpg', 'highway_mpg', 'price']

# horsepower，width，height, priceだけにする
auto2 = auto[['horsepower', 'width', 'height', 'price']]

# '?'をNaNに置換し，NaNがある行を削除
auto2 = auto2.replace('?', np.nan).dropna()
print(auto2.shape)

# horsepowerとpriceがobjectなので，数値型に変換
auto2 = auto2.assign(horsepower=pd.to_numeric(auto2.horsepower))
auto2 = auto2.assign(price=pd.to_numeric(auto2.price))


# データを説明変数Xと目的変数yに分ける
x = auto2.drop('price', axis=1)
y = auto2['price']

# 訓練データ，テストデータ分割
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.5, random_state=0)

# 通常の重回帰モデルの初期化と学習
mlr_model = LinearRegression()
mlr_model.fit(x_train, y_train)

# リッジ回帰とラッソ回帰について
# 基本的な構造は重回帰分析と同じ
# 違うのは回帰係数と切片を求める際に損失関数に正則化項を足して回帰式を推定する
# ラッソ回帰はモデルのパラメータのユークリッド距離、リッジ回帰はマンハッタン距離を足している。
# ラッソ回帰では値が0のパラメータが増え、リッジ回帰では値の小さいパラメータが増え、突出して大きな値が少なくなる。
# →何がどういいのか
# パラメータが多く複雑なモデルほど学習データにフィッティングしすぎてしまい、検証データ対する反応特性（汎化能力）が低下する
# リッジ回帰、ラッソ回帰では損失関数にパラメータに対するペナルティを付加する→モデルが複雑になりすぎないようにする
# そうすることで、学習データへの過学習を防ぐ

# alphaを増やす -> 正則化が強くなる -> モデルは簡潔になる
# alphaを減らす -> 正則化が弱くなる -> モデルは複雑になる
# リッジ回帰モデルの初期化と学習
ridge_model = Ridge(alpha=1000.0, random_state=0)
ridge_model.fit(x_train, y_train)

# ラッソ回帰モデルの初期化と学習
lasso_model = Lasso(alpha=1000.0, random_state=0)
lasso_model.fit(x_train, y_train)


# 構築したモデルのパラメータ（回帰係数と切片）の値
print('-------Linear Regression--------')
print(pd.Series(mlr_model.coef_, index=x.columns))
print(mlr_model.intercept_)
print('--------Ridge-------')
print(pd.Series(ridge_model.coef_, index=x.columns))
print(ridge_model.intercept_)
print('--------Lasso-------')
print(pd.Series(lasso_model.coef_, index=x.columns))
print(lasso_model.intercept_)


# 訓練データに対するスコア（決定係数）の算出
print('-------Linear Regression--------')
print(mlr_model.score(x_train, y_train))
print('--------Ridge-------')
print(ridge_model.score(x_train, y_train))
print('--------Lasso-------')
print(lasso_model.score(x_train, y_train))

# テストデータに対するスコア（決定係数）の算出
print('-------Linear Regression--------')
print(mlr_model.score(x_test, y_test))
print('--------Ridge-------')
print(ridge_model.score(x_test, y_test))
print('--------Lasso-------')
print(lasso_model.score(x_test, y_test))
