from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd

wine = load_wine()
x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)

# criterion="entropy"
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)
print(score)

# criterion="gini"
clf = tree.DecisionTreeClassifier(criterion="gini")
clf = clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)
print(score)

feature_name = ['酒精', '苹果酸', '灰', '灰的碱性', '镁', '总酚', '类黄酮', '非黄烷类酚类', '花青素', '颜色强度', '色调', 'od280/od315稀释葡萄酒', '脯氨酸']
print(*zip(feature_name, clf.feature_importances_))

clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=30)
clf = clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)  # 返回预测的准确度
print(score)

clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=30, splitter="random")
clf = clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)  # 返回预测的准确度
print(score)

clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=30, splitter="best")
clf = clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)  # 返回预测的准确度
print(score)

clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=30, splitter="random", max_depth=3,
                                  min_samples_leaf=10, min_samples_split=10)
clf = clf.fit(x_train, y_train)
print(clf.score(x_train, y_train))
print(clf.score(x_test, y_test))

import matplotlib.pyplot as plt

test = []
for i in range(10):
    clf = tree.DecisionTreeClassifier(max_depth=i + 1
                                      , criterion="entropy"
                                      , random_state=30
                                      , splitter="random")
    clf = clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    test.append(score)
plt.plot(range(1, 11), test, color="red", label="max_depth")
plt.legend()
plt.show()

print(clf.predict(x_test))


