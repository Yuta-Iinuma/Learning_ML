# データ加工・処理・分析ライブラリ
from sklearn.model_selection import cross_val_score  # 交差検証の便利な関数がある
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import numpy.random as random
import pandas as pd
import scipy as sp

# 機械学習ライブラリ
import sklearn

# 可視化ライブラリ
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# あやめのデータセット利用
from sklearn.datasets import load_iris

iris = load_iris()
x = iris.data  # 特徴量（入力，説明変数）
y = iris.target  # 目的変数（出力，クラス，カテゴリ）

# ホールドアウト法
# データセットから学習用データセットと検証用データセットに分割して検証する。
# 学習用データセットと検証用データセットをランダムにデータを分割し、それを複数回検証する
# 検証した複数のモデル評価の平均値をとって、最終的なモデル評価とする

# 問題点
# たまたま精度が高くなる場合がある。
# 限られたデータを分割するため、学習データを削られて、学習がうまくいかない場合がある。
# →簡単にいうと、学習用データセットのデータサイズが小さくなる。
# データの分散（ばらつき）が大きいと予測精度が低くなる
# →データ分類がランダムだから、データの分類の仕方により精度の評価に影響を及ぼす

# k-NNの例
# トレーニングとテストに分割して評価を繰り返し
# 今回ランダムデータ分割を1000回行う
results = []  # 結果のスコアを格納する空のリスト
for times in range(1000):
    # random_stateを指定しないとnp.randomを用いた乱数セットになる
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    # モデル初期化
    knn_model = KNeighborsClassifier(n_neighbors=1)
    # 学習
    knn_model.fit(x_train, y_train)
    # 結果をリストへ追加
    results.append(knn_model.score(x_test, y_test))

 # リストをnp.arrayに変換
results = np.array(results)
# print(results.mean())
# print(results.std())

# K分割交差検証法
# データをK個のブロックに分割
# そのうち１ブロックをデータの検証用にする
# それ以外のk-1ブロックのデータを学習用データにする
# それをK回繰り返し評価し、スコアの平均値や偏差を評価する
# ホールドアウト法では分割方法がランダムだったため、データのばらつきが大きいと、
# データの分類の仕方により精度の評価に影響を及ぼしていたが
# 交差検証ではブロックごとに分類しそれを複数回変えて検証するため、ばらつきの大きいデータでも使える


knn_model = KNeighborsClassifier(n_neighbors=1)  # モデル初期化

# 使うモデル，説明変数，目的変数，分割数を引数に入れるだけで，学習と評価を繰り返してスコアを算出
scores = cross_val_score(knn_model, x, y, cv=50)

print(scores.mean())
print(scores.std())

# 他のモデル検証方法
# leave one out法
# データから一つだけ要素を取り出しそれを検証用データにし、それ以外全てを学習用データセットにする。
# 例：データが100個あったら一個検証用、99個学習用
# これを複数回（上記の場合100回）繰り返し、スコアの平均や偏差を評価
# メリット：その他検証方法より学習用データにできる量が多いため、モデルの精度向上が見込める
# 問題点：一般的にスコアが高く出過ぎる傾向がある（過学習の疑い）、データ数が大きい場合計算処理時間が増大するため、学習コストが高い

# ブートストラップ法
# 実際に用意されたデータがN個あるとする
# そのデータから重複を許してN個のデータをランダムにとってくる
# 利点
# 分布を仮定しない解析手法
# 前提条件が皆無なため、いろいろな場面で使用できる
# デメリット
# ブートストラップ法の数少ない前提条件が、重複を許しランダムに抽出したデータが母集団だとみなすこと
# そのため、抽出したデータが本当に母集団を適切に反映していないと、真の母集団を反映できない
