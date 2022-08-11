# データ加工・処理・分析ライブラリ
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
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

import requests
import zipfile
import io

# データフレーム全部表示
pd.set_option('display.max_rows', None)

# 自動車価格データを取得
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
res = requests.get(url).content

# 取得したデータをDataFrameオブジェクトとして読み込み
auto = pd.read_csv(io.StringIO(res.decode('utf-8')), header=None)

# データの列にラベルを設定
auto.columns = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors',
                'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height',
                'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore',
                'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
# 必要なカラムだけ残す
auto2 = auto[['length', 'width', 'height',
              'engine-size', 'horsepower', 'price', 'make']]

# '?'をNaNに置換し，NaNがある行を削除
auto2 = auto2.replace('?', np.nan).dropna()

# makeカラムを２値にする．日本車メーカー:0，海外メーカー:1 とする場合
auto2['make'] = auto2['make'].map(lambda x: 1 if x == 'honda' or x == 'isuzu' or x ==
                                  'mazda' or x == 'mitsubishi' or x == 'nissan' or x == 'subaru' or x == 'toyota' else 0)

# horsepowerとpriceがobjectなので，数値型に変換
auto2 = auto2.assign(horsepower=pd.to_numeric(auto2.horsepower))
auto2 = auto2.assign(price=pd.to_numeric(auto2.price))

# データを説明変数Xと目的変数Yに分ける
X = auto2.drop('make', axis=1)
Y = auto2['make']

# 説明変数を標準化
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)

# 学習データ，検証データを半々にする
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)

# モデル用意
model = LogisticRegression()

# 学習データにモデルをフィット（学習）
model.fit(X_train, Y_train)

# 構築したモデルのパラメータ（回帰係数と残差）の値
print(model.coef_)
print('---------------')
print(model.intercept_)

# 訓練データに対するスコア（決定係数）の算出
result_train = model.score(X_train, Y_train)
print('---------------')
print(result_train)

# 構築したモデルで予測
Y_pred = model.predict(X_test)

# テストデータに対するスコア（決定係数）の算出
result_test = model.score(X_test, Y_test)
print(result_test)

# ロジスティック回帰分析は説明変数から2値の目的変数が起こる確率を説明・予測できる分析手法である。
# モデルの基本は重回帰と同じだが、重回帰で算出した値をロジスティック関数、シグモイド関数で変換することで2値の目的変数を予測
# 損失関数は交差エントロピー（クロスエントロピ）基準のもの
# 交差エントロピーは実際のデータと推定値の確率（確率分布）の間の近さを示す。
# 二乗誤差基準でも実装可能だが、交差エントロピーの方が効率が良い
