# -*- encoding: utf-8 -*-
# データ加工・処理・分析ライブラリ
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import numpy.random as random
from pandas import Series, DataFrame
import pandas as pd
import scipy as sp

# 機械学習ライブラリ
import sklearn

# 可視化ライブラリ
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
# matplotlib inline

# 小数第3位まで表示
# precision 3

import requests
import zipfile
import io

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

# 各カラムに「?」が何個あるかカウント
# print(auto2.isin(['?']).sum())

auto2 = auto2.replace('?', np.nan).dropna()

# 型をチェック
# print(auto2.dtypes)
# print('-------------------------')

# horsepowerとpriceがobjectなので，数値型に変換
auto2 = auto2.assign(horsepower=pd.to_numeric(auto2.horsepower))
auto2 = auto2.assign(price=pd.to_numeric(auto2.price))

# 型を再チェック
# print(auto2.dtypes)

# 相関を見てみる
# tmp = auto2.corr()
# print(tmp)

# データを 説明変数 X と 目的変数 y に分ける
x = auto2.drop('price', axis=1)
y = auto2['price']

# 重回帰分析
# 訓練データ，テストデータ分割
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.5, random_state=0)


# モデルの初期化と学習
mlr_model = LinearRegression()
mlr_model.fit(x_train, y_train)

# モデルの概要
# print(mlr_model)
# print('---------------')

# 構築したモデルのパラメータ（回帰係数と残差）の値
# 回帰係数：回帰分析時における説明変数の各要素がどの程度目的変数算出に影響しているかを示す。
# 今回は3要素（'horsepower', 'width', 'height'）がどの割合で予測対象に影響しているかを表す。
# print(pd.Series(mlr_model.coef_, index=x.columns))
# print('---------------')

# 残差実際のデータを用いて推定された回帰式から推定された数値と実際の数値との差
# 誤差とは違う
# 誤差は真の回帰式から推定された数値と実際の数値の差
# 真の回帰式は理論的なものであるため、誤差を計算で求めることはできない
# print(mlr_model.intercept_)

# 決定係数の算出：決定係数が1に近いと分析の精度が高い
# 予測のあてはまりの良さが決定係数なので、予測する内容や分析目的によっては決定係数が高くても意味をなさない
# 算出した回帰式と該当のデータがどの程度当てはまっているかを指す。
# →学習データより、testデータの方が決定係数がいいケースも考えられる。
# print(mlr_model.score(x_train, y_train))

# 構築したモデルで予測
y_pred = mlr_model.predict(x_test)

# 実際の値と予測値の確認
# print(y_test)
# print(y_pred)

# 訓練データに対するスコア（決定係数）の算出
result_train = mlr_model.score(x_train, y_train)
print(result_train)

# テストデータに対するスコア（決定係数）の算出
result_test = mlr_model.score(x_test, y_test)
print(result_test)
