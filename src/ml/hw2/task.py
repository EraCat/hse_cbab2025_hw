from typing import NoReturn

import numpy as np
from sklearn.neighbors import KDTree


# Task 1


class KMeans:
    def __init__(self, n_clusters: int, init: str = "k-means++", max_iter: int = 300):
        """
        
        Parameters
        ----------
        n_clusters : int
            Число итоговых кластеров при кластеризации.
        init : str
            Способ инициализации кластеров. Один из трех вариантов:
            1. random --- центроиды кластеров являются случайными точками,
            2. sample --- центроиды кластеров выбираются случайно из  X,
            3. k-means++ --- центроиды кластеров инициализируются 
                при помощи метода K-means++.
        max_iter : int
            Максимальное число итераций для kmeans.
        
        """
        self.centroids = None
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = 1e-4
        self.n_init = 50
        self.random_state = 42


    def _init_kmeans_pp(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n, d = X.shape
        k = self.n_clusters
        centroids = np.empty((k, d), dtype=float)

        idx0 = rng.integers(n)
        centroids[0] = X[idx0]

        dist2 = ((X - centroids[0]) ** 2).sum(axis=1)

        for j in range(1, k):
            total = dist2.sum()
            if total == 0.0:
                centroids[j:] = centroids[0]
                break

            probs = dist2 / total
            idx = rng.choice(n, p=probs)
            centroids[j] = X[idx]

            new_dist2 = ((X - centroids[j]) ** 2).sum(axis=1)
            dist2 = np.minimum(dist2, new_dist2)
            dist2[idx] = 0.0

        return centroids

    def _init_centroids(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n, d = X.shape

        if self.init == "sample":
            idx = rng.choice(n, size=self.n_clusters, replace=False)
            return X[idx].copy()

        if self.init == "random":
            mins = X.min(axis=0)
            maxs = X.max(axis=0)
            return rng.uniform(low=mins, high=maxs, size=(self.n_clusters, d))

        if self.init in ("k-means++", "kmeans++"):
            return self._init_kmeans_pp(X, rng)


    def fit(self, X: np.array, y=None) -> NoReturn:
        """
        Ищет и запоминает в self.centroids центроиды кластеров для X.

        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Неиспользуемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit обязаны принимать
            параметры X и y, даже если y не используется).

        """
        X = np.asarray(X, dtype=float)


        n, d = X.shape
        k = self.n_clusters

        base_rng = np.random.default_rng(self.random_state)

        best_centroids = None
        best_inertia = np.inf

        for _ in range(self.n_init):
            rng = np.random.default_rng(base_rng.integers(1_000_000_000))
            centroids = self._init_centroids(X, rng)
            prev_labels = None

            for _ in range(self.max_iter):
                dist2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
                labels = dist2.argmin(axis=1)

                if prev_labels is not None and np.array_equal(labels, prev_labels):
                    break
                prev_labels = labels

                new_centroids = np.empty((k, d), dtype=float)
                for j in range(k):
                    mask = (labels == j)
                    if mask.any():
                        new_centroids[j] = X[mask].mean(axis=0)
                    else:
                        farthest_i = dist2.min(axis=1).argmax()
                        new_centroids[j] = X[farthest_i]

                shift = np.linalg.norm(new_centroids - centroids)
                centroids = new_centroids
                if shift <= self.tol:
                    break

            dist2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            inertia = float(dist2.min(axis=1).sum())

            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids

        self.centroids = best_centroids


    def predict(self, X: np.array) -> np.array:
        """
        Для каждого элемента из X возвращает номер кластера,
        к которому относится данный элемент.

        Parameters
        ----------
        X : np.array
            Набор данных, для элементов которого находятся ближайшие кластера.

        Return
        ------
        labels : np.array
            Вектор индексов ближайших кластеров
            (по одному индексу для каждого элемента из X).

        """

        X = np.asarray(X, dtype=float)

        dist2 = ((X[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
        return dist2.argmin(axis=1)


# Task 2

class DBScan:
    def __init__(self, eps: float = 0.5, min_samples: int = 5,
                 leaf_size: int = 40, metric: str = "euclidean"):
        """

        Parameters
        ----------
        eps : float, min_samples : int
            Параметры для определения core samples.
            Core samples --- элементы, у которых в eps-окрестности есть
            хотя бы min_samples других точек.
        metric : str
            Метрика, используемая для вычисления расстояния между двумя точками.
            Один из трех вариантов:
            1. euclidean
            2. manhattan
            3. chebyshev
        leaf_size : int
            Минимальный размер листа для KDTree.

        """
        self.eps = eps
        self.min_samples = min_samples
        self.leaf_size = leaf_size
        self.metric = metric

    def _neighbors_all_kdtree(self, X: np.ndarray) -> list[np.ndarray]:
        tree = KDTree(X, leaf_size=self.leaf_size, metric=self.metric)
        ind = tree.query_radius(X, r=self.eps, return_distance=False)

        # query_radius включает саму точку; убираем её, чтобы соответствовать "других точек"
        neigh = []
        for i in range(X.shape[0]):
            arr = ind[i]
            if arr.size == 0:
                neigh.append(arr)
            else:
                neigh.append(arr[arr != i])
        return neigh

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Кластеризует элементы из X,
        для каждого возвращает индекс соотв. кластера.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit_predict обязаны принимать
            параметры X и y, даже если y не используется).
        Return
        ------
        labels : np.array
            Вектор индексов кластеров
            (Для каждой точки из X индекс соотв. кластера).

        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = X.astype(float, copy=False)

        n, d = X.shape
        labels = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool)

        neigh = self._neighbors_all_kdtree(X)

        is_core = np.array([len(neigh[i]) >= self.min_samples for i in range(n)], dtype=bool)

        cluster_id = 0

        for i in range(n):
            if visited[i]:
                continue

            visited[i] = True

            if not is_core[i]:
                continue

            labels[i] = cluster_id

            queue = list(neigh[i])
            in_queue = set(queue)

            while queue:
                j = queue.pop()

                if not visited[j]:
                    visited[j] = True

                    if is_core[j]:
                        for q in neigh[j]:
                            if q not in in_queue:
                                queue.append(q)
                                in_queue.add(q)

                if labels[j] == -1:
                    labels[j] = cluster_id

            cluster_id += 1

        return labels


# Task 3

class AgglomertiveClustering:
    def __init__(self, n_clusters: int = 16, linkage: str = "average"):
        """
        
        Parameters
        ----------
        n_clusters : int
            Количество кластеров, которые необходимо найти (то есть, кластеры 
            итеративно объединяются, пока их не станет n_clusters)
        linkage : str
            Способ для расчета расстояния между кластерами. Один из 3 вариантов:
            1. average --- среднее расстояние между всеми парами точек, 
               где одна принадлежит первому кластеру, а другая - второму.
            2. single --- минимальное из расстояний между всеми парами точек, 
               где одна принадлежит первому кластеру, а другая - второму.
            3. complete --- максимальное из расстояний между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
        """
        pass

    def fit_predict(self, X: np.array, y=None) -> np.array:
        """
        Кластеризует элементы из X, 
        для каждого возвращает индекс соотв. кластера.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit_predict обязаны принимать 
            параметры X и y, даже если y не используется).
        Return
        ------
        labels : np.array
            Вектор индексов кластеров
            (Для каждой точки из X индекс соотв. кластера).

        """
        pass
