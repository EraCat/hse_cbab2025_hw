from sklearn.model_selection import train_test_split
import numpy as np
import pandas
import random
import copy
from catboost import CatBoostClassifier
from typing import Callable


# Task 0

def gini(x: np.ndarray) -> float:
    """
    Считает коэффициент Джини для массива меток x.
    """
    if len(x) == 0:
        return 0.0

    u, counts = np.unique(x, return_counts=True)
    probs = counts / counts.sum()
    return 1.0 - np.sum(probs ** 2)

def entropy(x: np.ndarray) -> float:
    """
    Считает энтропию для массива меток x.
    """
    u, counts = np.unique(x, return_counts=True)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def gain(left_y: np.ndarray, right_y: np.ndarray, criterion: Callable) -> float:
    """
    Считает информативность разбиения массива меток.

    Parameters
    ----------
    left_y : np.ndarray
        Левая часть разбиения.
    right_y : np.ndarray
        Правая часть разбиения.
    criterion : Callable
        Критерий разбиения.
    """
    total = len(left_y) + len(right_y)
    if total == 0:
        return 0.0
    parent_y = np.concatenate([left_y, right_y])
    parent = criterion(parent_y)

    left_w = len(left_y) / total
    right_w = len(right_y) / total

    current = left_w * criterion(left_y) + right_w * criterion(right_y)
    return parent - current


# Task 1
class _DecisionTreeLeaf:
    def __init__(self, y):
        values, counts = np.unique(y, return_counts=True)
        self.label = values[np.argmax(counts)]


class _DecisionTreeNode:
    def __init__(self, feature, left, right):
        self.feature = feature
        self.left = left
        self.right = right

class DecisionTree:
    def __init__(self, X, y, criterion="gini", max_depth=None, min_samples_leaf=1, max_features="auto"):
        X = np.asarray(X)
        y = np.asarray(y)

        self.criterion_name = criterion
        if criterion == "gini":
            self.criterion = gini
        elif criterion == "entropy":
            self.criterion = entropy

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_features_ = X.shape[1]

        n = len(X)
        bootstrap_indices = np.random.choice(n, size=n, replace=True)
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[np.unique(bootstrap_indices)] = False

        self.X_bootstrap = X[bootstrap_indices]
        self.y_bootstrap = y[bootstrap_indices]
        self.X_oob = X[oob_mask]
        self.y_oob = y[oob_mask]

        self.root = self._build_tree(self.X_bootstrap, self.y_bootstrap, depth=0)


    def _build_tree(self, X, y, depth):
        if len(y) == 0:
            return None

        # если все объекты одного класса
        if np.all(y == y[0]):
            return _DecisionTreeLeaf(y)

        # ограничения
        if self.max_depth is not None and depth >= self.max_depth:
            return _DecisionTreeLeaf(y)

        if len(y) < 2 * self.min_samples_leaf:
            return _DecisionTreeLeaf(y)

        split_feature = self._find_best_split(X, y)

        if split_feature is None:
            return _DecisionTreeLeaf(y)

        left_mask = X[:, split_feature] == 0
        right_mask = X[:, split_feature] == 1

        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            return _DecisionTreeLeaf(y)

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return _DecisionTreeNode(split_feature, left_subtree, right_subtree)

    def _find_best_split(self, X, y):
        feature_count = self._get_max_features()
        candidate_features = np.random.choice(
            self.n_features_,
            size=feature_count,
            replace=False
        )

        best_gain = -1.0
        best_feature = None

        for feature in candidate_features:
            left_mask = X[:, feature] == 0
            right_mask = X[:, feature] == 1

            if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                continue

            cur_gain = gain(y[left_mask], y[right_mask], self.criterion)

            if cur_gain > best_gain:
                best_gain = cur_gain
                best_feature = feature

        return best_feature

    def _get_max_features(self):
        if self.max_features == "auto":
            return max(1, int(np.sqrt(self.n_features_)))
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(self.n_features_)))
        if self.max_features == "log2":
            return max(1, int(np.log2(self.n_features_)))
        if self.max_features is None:
            return self.n_features_
        if isinstance(self.max_features, int):
            return max(1, min(self.n_features_, self.max_features))
        if isinstance(self.max_features, float):
            return max(1, min(self.n_features_, int(self.max_features * self.n_features_)))


    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_one(x, self.root) for x in X])

    def _predict_one(self, x, node):
        if isinstance(node, _DecisionTreeLeaf):
            return node.label

        if x[node.feature] == 0:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)


# Task 2

class RandomForestClassifier:
    def __init__(self, criterion="gini", max_depth=None, min_samples_leaf=1, max_features="auto", n_estimators=10):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.trees = []
        self.n_features_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.n_features_ = X.shape[1]
        self.trees = []

        for _ in range(self.n_estimators):
            tree = DecisionTree(
                X, y,
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features
            )
            self.trees.append(tree)

    def predict(self, X):
        X = np.asarray(X)

        all_preds = np.array([tree.predict(X) for tree in self.trees])

        result = []
        for i in range(X.shape[0]):
            values, counts = np.unique(all_preds[:, i], return_counts=True)
            result.append(values[np.argmax(counts)])

        return np.array(result)

# Task 3
def feature_importance(rfc):
    n_features = rfc.n_features_
    importance = np.zeros(n_features, dtype=float)
    used_trees = 0

    for tree in rfc.trees:
        X_oob = tree.X_oob
        y_oob = tree.y_oob

        if len(y_oob) == 0:
            continue

        base_pred = tree.predict(X_oob)
        err_oob = np.mean(base_pred != y_oob)

        tree_importance = np.zeros(n_features, dtype=float)

        for j in range(n_features):
            X_perm = X_oob.copy()
            perm = np.random.permutation(len(X_perm))
            X_perm[:, j] = X_perm[perm, j]

            perm_pred = tree.predict(X_perm)
            err_oob_j = np.mean(perm_pred != y_oob)

            tree_importance[j] = err_oob_j - err_oob

        importance += tree_importance
        used_trees += 1

    if used_trees == 0:
        return importance

    return importance / used_trees

# Task 4
rfc_age = RandomForestClassifier(
    criterion="gini",
    max_depth=12,
    min_samples_leaf=1,
    max_features='0.4',
    n_estimators=100
)

rfc_gender = RandomForestClassifier(
    criterion="gini",
    max_depth=10,
    min_samples_leaf=5,
    max_features="auto",
    n_estimators=100
)


# Task 5
# Здесь нужно загрузить уже обученную модели
# https://catboost.ai/en/docs/concepts/python-reference_catboost_save_model
# https://catboost.ai/en/docs/concepts/python-reference_catboost_load_model

catboost_rfc_age = CatBoostClassifier()
catboost_rfc_age.load_model(__file__[:-7] + "catboost_rfc_age.cbm")

catboost_rfc_gender = CatBoostClassifier()
catboost_rfc_gender.load_model(__file__[:-7] + "catboost_rfc_gender.cbm")