# データ加工・処理・分析ライブラリ
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import numpy as np
import numpy.random as random
from pandas import Series, DataFrame
import pandas as pd

# 機械学習ライブラリ
import sklearn

# 可視化ライブラリ
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from sklearn.datasets import make_blobs

# 分類用データセットの作成
X, _ = make_blobs(random_state=2)
# plt.scatter(X[:, 0], X[:, 1], color='orange')

k = 3

# k-means法
# 与えられたデータをK個のクラスタに分類する
# フロー
# いくつのクラスターに分類するか決める
# 各クラスターに適当に分類したいデータに振り分ける
# 各クラスターごとに振り分けたデータの平均ベクトルを求める＊
# 平均ベクトルを各クラスターの代表ベクトルとする
# 代表ベクトルと分類したデータとの距離を測る
# 各データが分類されたクラスターの代表ベクトルより近い代表ベクトルがあれば、そのデータを一番近い代表ベクトルのクラスタに振り分け直す
# 振り分け直す必要があれば再度＊から実行、なければ終了
kmeans_model = KMeans(n_clusters=k, init='random')
kmeans_model.fit(X)

# k-means++：初期重心点をなるべく広げる．k-meansより安定的な結果を得られる．
kmeans_pp_model = KMeans(n_clusters=k)
kmeans_pp_model.fit(X)


# クラスター数の算出
# Kの値を順に変化させていき、傾向を確認する
dist = []  # 距離の総和を格納する変数
for k in range(1, 15):
    model = KMeans(n_clusters=k, init='random', random_state=0)
    model.fit(X)
    dist.append(model.inertia_)

# グラフ描画
# plt.plot(range(1, 15), dist, marker='.')
# plt.xlabel('No. of Clusters')
# plt.ylabel('Distortion')

# 階層型クラスタリング
# 事前にクラスター数の設定必要
# データセットの用意
# 100サンプルのデータをガウス分布に従って生成（中心は3点, 特徴量は2次元）
X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=10)
# plt.scatter(X[:,0], X[:,1], color='orange')

# DEBSCAN
# クラスター数の算出自動


# DBSCANモデルの作成と学習
dbs_model = DBSCAN(eps=3, min_samples=10)
data = pd.read_excel('animals.xlsx')
dbs_model.fit(data.iloc[:, 1:])

# クラスタ番号を取得
y_pred = dbs_model.fit_predict(data.iloc[:, 1:])

# クラスタリング結果の描画のためにデータフレーム作成
result_data = pd.concat(
    [pd.DataFrame(X[:, 0]), pd.DataFrame(X[:, 1]), pd.DataFrame(y_pred)], axis=1)
result_data.columns = ['feature1', 'feature2', 'cluster']
