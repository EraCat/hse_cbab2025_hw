from typing import Callable, Union, Optional, Dict, Any, List
import numpy as np


def gini(x: np.ndarray) -> float:
    """
    Считает коэффициент Джини для массива меток x.
    """
    x = np.asarray(x)
    if x.size == 0:
        return 0.0

    _, counts = np.unique(x, return_counts=True)
    probs = counts / counts.sum()
    return float(1.0 - np.sum(probs ** 2))



def entropy(x: np.ndarray) -> float:
    """
    Считает энтропию для массива меток x.
    """
    x = np.asarray(x)
    if x.size == 0:
        return 0.0

    _, counts = np.unique(x, return_counts=True)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))



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
    left_y = np.asarray(left_y)
    right_y = np.asarray(right_y)
    total = left_y.size + right_y.size
    if total == 0:
        return 0.0

    parent_y = np.concatenate([left_y, right_y])
    parent_impurity = criterion(parent_y)
    children_impurity = (
        (left_y.size / total) * criterion(left_y)
        + (right_y.size / total) * criterion(right_y)
    )
    return float(parent_impurity - children_impurity)


class DecisionTreeLeaf:
    """

    Attributes
    ----------
    y : Тип метки (напр., int или str)
        Метка класса, который встречается чаще всего среди элементов листа дерева
    """
    def __init__(self, ys):
        ys = np.asarray(ys)
        values, counts = np.unique(ys, return_counts=True)
        best_idx = int(np.argmax(counts))
        self.y = values[best_idx]
        total = counts.sum()
        self.proba = {label: float(cnt / total) for label, cnt in zip(values, counts)}


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
    def __init__(self, split_dim: int, split_value: float,
                 left: Union['DecisionTreeNode', DecisionTreeLeaf],
                 right: Union['DecisionTreeNode', DecisionTreeLeaf]):
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right


class DecisionTreeClassifier:
    """
    Attributes
    ----------
    root : Union[DecisionTreeNode, DecisionTreeLeaf]
        Корень дерева.

    (можете добавлять в класс другие аттрибуты).

    """
    def __init__(self, criterion: str = "gini",
                 max_depth: Optional[int] = None,
                 min_samples_leaf: int = 1):
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
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")

        self.criterion_name = criterion
        self.criterion = gini if criterion == "gini" else entropy
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.classes_ = None

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
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")
        if len(y) == 0:
            raise ValueError("training set must be non-empty")

        self.classes_ = np.unique(y)
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int):
        if len(np.unique(y)) == 1:
            return DecisionTreeLeaf(y)

        if self.max_depth is not None and depth >= self.max_depth:
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
        best_gain = -np.inf
        best_split = None

        for split_dim in range(n_features):
            values = np.unique(X[:, split_dim])
            if values.size < 2:
                continue

            thresholds = (values[:-1] + values[1:]) / 2.0
            for split_value in thresholds:
                left_mask = X[:, split_dim] < split_value
                right_mask = ~left_mask
                left_count = int(np.sum(left_mask))
                right_count = n_samples - left_count

                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue

                cur_gain = gain(y[left_mask], y[right_mask], self.criterion)
                if cur_gain > best_gain:
                    best_gain = cur_gain
                    best_split = (split_dim, float(split_value))

        if best_split is None or best_gain <= 0:
            return None
        return best_split

    def _predict_one_proba(self, x: np.ndarray) -> Dict[Any, float]:
        node = self.root
        while isinstance(node, DecisionTreeNode):
            if x[node.split_dim] < node.split_value:
                node = node.left
            else:
                node = node.right

        result = {label: 0.0 for label in self.classes_}
        for label, prob in node.proba.items():
            result[label] = prob
        return result

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
        if self.root is None:
            raise ValueError("The tree is not fitted yet")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return [self._predict_one_proba(row) for row in X]

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
