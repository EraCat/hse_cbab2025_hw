from sklearn.datasets import make_blobs, make_moons
import numpy as np
import pandas
import random
from typing import Callable, Union, Optional, Dict, Any, List

# Task 1


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


# Task 2


class DecisionTreeLeaf:
    """

    Attributes
    ----------
    y : Тип метки (напр., int или str)
        Метка класса, который встречается чаще всего среди элементов листа дерева
    """

    def __init__(self, ys):
        values, counts = np.unique(ys, return_counts=True)
        self.y = values[np.argmax(counts)]
        total = counts.sum()
        self.proba = {cls: cnt / total for cls, cnt in zip(values, counts)}



class DecisionTreeNode:
    """

    Attributes
    ----------
    split_dim : int
        Измерение, по которому разбиваем выборку.
    split_value : float
        Значение, по которому разбираем выборку.
    left : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] < split_value.
    right : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] >= split_value.
    """

    def __init__(
        self,
        split_dim: int,
        split_value: float,
        left: Union["DecisionTreeNode", DecisionTreeLeaf],
        right: Union["DecisionTreeNode", DecisionTreeLeaf],
    ):
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right


# Task 3


class DecisionTreeClassifier:
    """
    Attributes
    ----------
    root : Union[DecisionTreeNode, DecisionTreeLeaf]
        Корень дерева.

    (можете добавлять в класс другие аттрибуты).

    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
    ):
        """
        Parameters
        ----------
        criterion : str
            Задает критерий, который будет использоваться при построении дерева.
            Возможные значения: "gini", "entropy".
        max_depth : Optional[int]
            Ограничение глубины дерева. Если None - глубина не ограничена.
        min_samples_leaf : int
            Минимальное количество элементов в каждом листе дерева.

        """
        self.root = None
        self.criterion_name = criterion
        self.criterion = gini if criterion == "gini" else entropy
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.clss = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Строит дерево решений по обучающей выборке.

        Parameters
        ----------
        X : np.ndarray
            Обучающая выборка.
        y : np.ndarray
            Вектор меток классов.
        """
        np.asarray(X)
        self.clss = np.unique(y)
        self.root = self._build_tree(X, y, 0)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int):
        if len(np.unique(y)) == 1:
            return DecisionTreeLeaf(y)

        if depth == self.max_depth:
            return DecisionTreeLeaf(y)

        if len(y) < 2 * self.min_samples_leaf:
            return DecisionTreeLeaf(y)

        best_split = self._find_best_split(X, y)
        if best_split is None:
            return DecisionTreeLeaf(y)

        split_dim, split_value = best_split
        left_mask = X[:, split_dim] < split_value
        right_mask = ~left_mask
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return DecisionTreeNode(split_dim, split_value, left_subtree, right_subtree)




    def _find_best_split(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        n_classes = len(self.clss)

        total_counts = np.bincount(y, minlength=n_classes)
        parent_impurity = self._impurity_from_counts(total_counts)

        best_gain = -np.inf
        best_split = None

        for dim in range(n_features):
            order = np.argsort(X[:, dim], kind="mergesort")
            x_sorted = X[order, dim]
            y_sorted = y[order]

            left_counts = np.zeros(n_classes, dtype=np.int64)
            right_counts = total_counts.copy()

            for i in range(n_samples - 1):
                cls = y_sorted[i]
                left_counts[cls] += 1
                right_counts[cls] -= 1

                if x_sorted[i] == x_sorted[i + 1]:
                    continue

                left_size = i + 1
                right_size = n_samples - left_size

                if left_size < self.min_samples_leaf or right_size < self.min_samples_leaf:
                    continue

                left_imp = self._impurity_from_counts(left_counts)
                right_imp = self._impurity_from_counts(right_counts)

                cur_gain = (
                    parent_impurity
                    - (left_size / n_samples) * left_imp
                    - (right_size / n_samples) * right_imp
                )

                if cur_gain > best_gain:
                    best_gain = cur_gain
                    best_split = (dim, float((x_sorted[i] + x_sorted[i + 1]) / 2.0))

        if best_split is None or best_gain <= 0:
            return None
        return best_split

    def _impurity_from_counts(self, counts: np.ndarray) -> float:
        total = counts.sum()
        if total == 0:
            return 0.0

        probs = counts[counts > 0] / total

        if self.criterion_name == "gini":
            return float(1.0 - np.sum(probs ** 2))
        else:
            return float(-np.sum(probs * np.log2(probs)))

    def predict_proba(self, X: np.ndarray) -> List[Dict[Any, float]]:
        """
        Предсказывает вероятность классов для элементов из X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.

        Return
        ------
        List[Dict[Any, float]]
            Для каждого элемента из X возвращает словарь
            {метка класса -> вероятность класса}.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return [self._predict_one_proba(row) for row in X]

    def _predict_one_proba(self, x: np.ndarray) -> Dict[Any, float]:
        node = self.root
        while isinstance(node, DecisionTreeNode):
            if x[node.split_dim] < node.split_value:
                node = node.left
            else:
                node = node.right

        result = {label: 0.0 for label in self.clss}
        for label, prob in node.proba.items():
            result[label] = prob
        return result

    def predict(self, X: np.ndarray) -> list:
        """
        Предсказывает классы для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.

        Return
        ------
        list
            Вектор предсказанных меток для элементов X.
        """
        proba = self.predict_proba(X)
        return [max(p.keys(), key=lambda k: p[k]) for p in proba]


# Task 4
task4_dtc = DecisionTreeClassifier(criterion="entropy", max_depth=6, min_samples_leaf=1)

