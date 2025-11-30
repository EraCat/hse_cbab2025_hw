import heapq
from dataclasses import dataclass
from typing import NoReturn, Tuple, List

import numpy as np
import pandas as pd


# Task 1

def read_cancer_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """

    Parameters
    ----------
    path_to_csv : str
        Путь к cancer датасету.

    Returns
    -------
    X : np.array
        Матрица признаков опухолей.
    y : np.array
        Вектор бинарных меток, 1 соответствует доброкачественной опухоли (M),
        0 --- злокачественной (B).

    """

    df: pd.DataFrame = pd.read_csv(path_to_csv)

    df: pd.DataFrame = df.sample(frac=1, random_state=42)

    x = df.iloc[:, 1:].to_numpy()
    y = df.iloc[:, 0].map({'M': 1, 'B': 0}).to_numpy()

    return x, y


def read_spam_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """
     
    Parameters
    ----------
    path_to_csv : str
        Путь к spam датасету.

    Returns
    -------
    X : np.array
        Матрица признаков сообщений.
    y : np.array
        Вектор бинарных меток, 
        1 если сообщение содержит спам, 0 если не содержит.
    
    """

    df = pd.read_csv(path_to_csv)
    df: pd.DataFrame = df.sample(frac=1, random_state=42)

    x = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()

    return x, y


# Task 2
def train_test_split(X: np.array, y: np.array, ratio: float) -> Tuple[np.array, np.array, np.array, np.array]:
    """

    Parameters
    ----------
    X : np.array
        Матрица признаков.
    y : np.array
        Вектор меток.
    ratio : float
        Коэффициент разделения.

    Returns
    -------
    X_train : np.array
        Матрица признаков для train выборки.
    y_train : np.array
        Вектор меток для train выборки.
    X_test : np.array
        Матрица признаков для test выборки.
    y_test : np.array
        Вектор меток для test выборки.

    """
    split_index = int(len(X) * ratio)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, y_train, X_test, y_test


# Task 3
def get_precision_recall_accuracy(y_pred: np.array, y_true: np.array) -> Tuple[np.array, np.array, float]:
    """

    Parameters
    ----------
    y_pred : np.array
        Вектор классов, предсказанных моделью.
    y_true : np.array
        Вектор истинных классов.

    Returns
    -------
    precision : np.array
        Precision = TP / (TP + FP)
        Вектор с precision для каждого класса.
    recall : np.array
        Recall = TP / (TP + FN)
        Вектор с recall для каждого класса.
    accuracy : float
        Значение метрики accuracy (одно для всех классов).

    """
    classes = np.unique(y_true)
    precision = np.zeros(len(classes))
    recall = np.zeros(len(classes))

    for class_id in classes:
        tp = np.sum((y_pred == class_id) & (y_true == class_id))
        fp = np.sum((y_pred == class_id) & (y_true != class_id))
        fn = np.sum((y_pred != class_id) & (y_true == class_id))

        precision[class_id] = tp / (tp + fp) if tp + fp != 0 else 0
        recall[class_id] = tp / (tp + fn) if tp + fn != 0 else 0

    accuracy = np.sum(y_pred == y_true) / len(y_true)

    return precision, recall, accuracy


# Task 4

@dataclass(slots=True)
class KDNode:
    axis: int = None
    split: float = None
    left: 'KDNode' = None
    right: 'KDNode' = None

    idxs: np.ndarray = None

    @property
    def is_leaf(self) -> bool:
        return self.idxs is not None


class KDTree:
    def __init__(self, X: np.ndarray, leaf_size: int = 40):
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которому строится дерево.
        leaf_size : int
            Минимальный размер листа
            (то есть, пока возможно, пространство разбивается на области,
            в которых не меньше leaf_size точек).

        Returns
        -------

        """
        self.X = X
        self.leaf_size = leaf_size
        # self.n_features - размерность пространства
        self.n_features = X.shape[1]

        # Создаем массив индексов от 0 до N-1
        idxs = np.arange(X.shape[0])

        # Запускаем рекурсивное построение
        self.root = self._build_tree(idxs, 0)

    def _build_tree(self, idxs: np.ndarray, depth: int) -> KDNode:
        m = idxs.size

        if m < self.leaf_size * 2:
            return KDNode(idxs=idxs)

        axis = depth % self.n_features
        mid = m // 2

        order = np.argsort(self.X[idxs, axis])
        idxs_sorted = idxs[order]

        split = float(self.X[idxs_sorted[mid], axis])

        left_idxs = idxs_sorted[:mid]
        right_idxs = idxs_sorted[mid:]

        left = self._build_tree(left_idxs, depth + 1)
        right = self._build_tree(right_idxs, depth + 1)

        return KDNode(axis=axis, split=split, left=left, right=right)



    def query(self, X: np.ndarray, k: int = 1) -> List[List[int]]:
        Xq = np.asarray(X, dtype=float)           # запросы тоже приводим к float
        if Xq.ndim != 2 or Xq.shape[1] != self.n_features:  # проверяем размерность (n_queries, d)
            pass

        n_train = self.X.shape[0]                 # количество точек в дереве (обучающая выборка)
        k = int(k)                                # k приводим к int
        if k <= 0:                                # k должно быть положительным
            pass
        k = min(k, n_train)                       # нельзя попросить больше соседей, чем точек всего

        results: List[List[int]] = []             # сюда будем складывать ответы для каждого запроса

        for q in Xq:                              # для каждого запроса q (q — вектор размера d)
            heap: List[tuple[float, int]] = []    # куча для хранения лучших кандидатов: (-dist2, idx)

            # Почему (-dist2, idx):
            # heapq — это min-heap, а нам удобно быстро знать ХУДШЕГО из текущих k (максимальную dist2),
            # поэтому кладём отрицание: самый большой dist2 даст самый маленький (-dist2) и окажется в heap[0].

            def push(idx: int, d2: float) -> None:        # пытаемся добавить кандидата (idx, dist^2)
                if len(heap) < k:                         # если ещё не набрали k кандидатов
                    heapq.heappush(heap, (-d2, idx))      # просто добавляем
                else:                                     # если уже есть k
                    if d2 < -heap[0][0]:                  # сравниваем с худшим из k (это -heap[0][0])
                        heapq.heapreplace(heap, (-d2, idx))  # заменяем худшего новым более близким

            def worst_d2() -> float:                      # текущий "порог отсечения"
                return float("inf") if len(heap) < k else -heap[0][0]  # если кандидатов мало — порога нет

            def rec(node: KDNode) -> None:                # рекурсивный обход дерева
                if node.is_leaf:                          # если мы в листе
                    pts = self.X[node.idxs]               # берём реальные точки из self.X по индексам листа
                    d2s = np.sum((pts - q) ** 2, axis=1)  # считаем евклидово расстояние^2 до всех точек листа
                    for idx_i, d2 in zip(node.idxs, d2s): # перебираем точки листа и их d2
                        push(int(idx_i), float(d2))       # пытаемся добавить в топ-k
                    return                                # лист обработан — выходим

                axis = node.axis                          # ось разделения в этом узле
                split = node.split                        # порог разделения в этом узле

                # Выбираем, куда идти сначала (в "ближнюю" ветку):
                if q[axis] < split:                       # если по axis запрос слева от split
                    near, far = node.left, node.right     # сначала идём влево, потом (возможно) вправо
                else:                                     # иначе запрос справа
                    near, far = node.right, node.left     # сначала вправо, потом (возможно) влево

                rec(near)                                 # сначала обрабатываем ближнюю ветку

                # Теперь решаем: нужно ли вообще лезть в дальнюю ветку?
                # Идея отсечения:
                # расстояние от q до плоскости разреза по axis равно |q[axis] - split|.
                # Если оно уже больше текущего худшего из k лучших, то в дальнем поддереве
                # не может быть точек ближе (для евклидовой метрики).
                diff = float(q[axis] - split)             # разница по оси axis до границы split
                if diff * diff < worst_d2():              # если плоскость достаточно близко — дальняя ветка может улучшить ответ
                    rec(far)                              # тогда проверяем дальнюю ветку

            rec(self.root)                                # запускаем поиск для одного запроса от корня

            # heap хранит k лучших, но не отсортированных по возрастанию расстояния:
            best = [(-neg_d2, idx) for (neg_d2, idx) in heap]  # превращаем в (dist2, idx)
            best.sort(key=lambda t: t[0])                 # сортируем по dist2 (чтобы ответ был аккуратный)
            results.append([idx for _, idx in best])      # сохраняем только индексы соседей (как требует ТЗ)

        return results


# Task 5

class KNearest:
    def __init__(self, n_neighbors: int = 5, leaf_size: int = 30):
        """

        Parameters
        ----------
        n_neighbors : int
            Число соседей, по которым предсказывается класс.
        leaf_size : int
            Минимальный размер листа в KD-дереве.

        """
        pass

    def fit(self, X: np.array, y: np.array) -> NoReturn:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которым строится классификатор.
        y : np.array
            Метки точек, по которым строится классификатор.

        """
        pass

    def predict_proba(self, X: np.array) -> List[np.array]:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.
        
        Returns
        -------
        list[np.array]
            Список np.array (длина каждого np.array равна числу классов):
            вероятности классов для каждой точки X.
            

        """

        pass

    def predict(self, X: np.array) -> np.array:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.
        
        Returns
        -------
        np.array
            Вектор предсказанных классов.
            

        """
        return np.argmax(self.predict_proba(X), axis=1)


def main():
    read_cancer_dataset('cancer.csv')
    read_spam_dataset('spam.csv')


if __name__ == "__main__":
    main()
