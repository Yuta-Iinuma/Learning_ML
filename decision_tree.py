from sklearn.tree import plot_tree
from IPython.display import Image
import pydotplus
from sklearn import tree
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import requests
import io
import matplotlib.pyplot as plt

# データを取得
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
res = requests.get(url).content

# 取得したデータをDataFrameオブジェクトとして読み込み
mushroom = pd.read_csv(io.StringIO(res.decode('utf-8')), header=None)

mushroom.columns = ['classes', 'cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor', 'gill_attachment',
                    'gill_spacing', 'gill_size', 'gill_color', 'stalk_shape', 'stalk_root', 'stalk_surface_above_ring',
                    'stalk_surface_below_ring', 'stalk_color_above_ring', 'stalk_color_below_ring', 'veil_type',
                    'veil_color', 'ring_number', 'ring_type', 'spore_print_color', 'population', 'habitat']
#説明変数をgill-color, gill-attachment, odor, cap-colorの４つに限定
mushroom2 = mushroom[['gill_color', 'gill_attachment', 'odor', 'cap_color']]

# カテゴリデータなのでダミー変数（one-hotベクトル）にする
# →文字列でカテゴリー分けされたデータを数値変換する
# ex)高い、低い→0,1
mushroom_dummy = pd.get_dummies(mushroom2)

# 目的変数も整理（p（毒キノコ）なら1，e（食用）なら0とする）
classes2 = mushroom['classes'].map(lambda x: 1 if x == 'p' else 0)

# mushroom_dummyに追加
mushroom_dummy['classes'] = classes2


# 説明変数と目的変数
x = mushroom_dummy.drop('classes', axis=1)
y = mushroom_dummy['classes']

# 訓練データとテストデータに分ける
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)


# 決定木とは
# 各説明変数に対する条件分岐ルールを断層的に繰り返し、目的変数を推定するモデル
# 目的変数がカテゴリー：分類木、目的変数が数値：回帰木
# 決定木の条件分岐のルール
# 多数の説明変数の中から最も有益な説明変数の条件を選び、順番に処理していく
# 不純度：説明変数の条件が識別に使えるか否かを評価する指標
# エントロピーとジニ不純度がある
# ジニ不純度は、データを分類したときに、各カテゴリーにどれだけ不純なデータ（どれだけ分類が間違っている要素）があるかを評価
# エントロピーは複雑性の認識でOK、各カテゴリーの複雑性を評価し不純度が低ければ複雑性も低くなる

# DecisionTreeClassifier()の引数'criterion'で学習データの分類の際に使用する不純度の算出方法を指定

# 決定木モデルの初期化と学習
tree_model = DecisionTreeClassifier(
    criterion='entropy', max_depth=5, random_state=0)
tree_model.fit(x_train, y_train)


# 訓練データの識別率
result = tree_model.score(x_train, y_train)
print(result)

# テストデータの識別率
result = tree_model.score(x_test, y_test)
print(result)

# plt.figure(figsize=(15, 10))
plot_tree(tree_model, feature_names=x.columns, class_names=True, filled=True)
plt.show()
