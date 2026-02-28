import numpy as np
import copy
from cvxopt import spmatrix, matrix, solvers
from sklearn.datasets import make_classification, make_moons, make_blobs
from typing import NoReturn, Callable

solvers.options['show_progress'] = False

# Task 1

class LinearSVM:
    def __init__(self, C: float):
        """

        Parameters
        ----------
        C : float
            Soft margin coefficient.

        """
        self.C = C
        self.w = None
        self.b = None
        self.support = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает SVM, решая задачу оптимизации при помощи cvxopt.solvers.qp

        Parameters
        ----------
        X : np.ndarray
            Данные для обучения SVM.
        y : np.ndarray
            Бинарные метки классов для элементов X
            (можно считать, что равны -1 или 1).

        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)

        n, d = X.shape
        y = np.where(y > 0, 1.0, -1.0)
        K =X @ X.T

        P = (np.outer(y, y) * K)
        q = -np.ones(n)
        G = np.vstack([-np.eye(n), np.eye(n)])
        h = np.hstack([np.zeros(n), self.C*np.ones(n)])

        A = y.reshape(1, -1)
        b = np.array([0.0])

        P_cvx = matrix(P, tc='d')
        q_cvx = matrix(q, tc='d')
        G_cvx = matrix(G, tc='d')
        h_cvx = matrix(h, tc='d')
        A_cvx = matrix(A, tc='d')
        b_cvx = matrix(b, tc='d')

        sol = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx, A_cvx, b_cvx)
        alpha = np.array(sol['x'], dtype=np.float64).reshape(-1)

        eps = 1e-6
        support = np.where(alpha > eps)[0]
        self.support = support

        self.w = (X.T @ (alpha * y)).reshape(-1)

        free = np.where((alpha > eps) & (alpha < self.C - eps))[0]

        if free.size > 0:
            self.b = float(np.mean(y[free] - X[free] @ self.w))
        else:
            if support.size == 0:
                self.b = 0.0
            else:
                self.b = float(np.mean(y[support] - X[support] @ self.w))




    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает значение решающей функции.

        Parameters
        ----------
        X : np.ndarray
            Данные, для которых нужно посчитать значение решающей функции.

        Return
        ------
        np.ndarray
            Значение решающей функции для каждого элемента X
            (т.е. то число, от которого берем знак с целью узнать класс).

        """
        X = np.asarray(X, dtype=np.float64)
        return X @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Классифицирует элементы X.

        Parameters
        ----------
        X : np.ndarray
            Данные, которые нужно классифицировать

        Return
        ------
        np.ndarray
            Метка класса для каждого элемента X.

        """
        return np.sign(self.decision_function(X))

# Task 2

def get_polynomial_kernel(c=1, power=2):
    "Возвращает полиномиальное ядро с заданной константой и степенью"

    def kernel(X: np.ndarray, x: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64).reshape(-1)

        dots = X @ x
        return (dots + c) ** power

    return kernel

def get_gaussian_kernel(sigma=1.):
    "Возвращает ядро Гаусса с заданным коэффицинтом сигма"

    def kernel(X: np.ndarray, x: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64).reshape(1, -1)

        diff = X - x                       # (n, m)
        dist2 = np.sum(diff * diff, axis=1) # (n,)
        return np.exp(-sigma * dist2)

    return kernel

# Task 3

class KernelSVM:
    def __init__(self, C: float, kernel: Callable):
        """

        Parameters
        ----------
        C : float
            Soft margin coefficient.
        kernel : Callable
            Функция ядра.

        """
        self.C = C
        self.kernel = kernel
        self.support = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает SVM, решая задачу оптимизации при помощи cvxopt.solvers.qp

        Parameters
        ----------
        X : np.ndarray
            Данные для обучения SVM.
        y : np.ndarray
            Бинарные метки классов для элементов X
            (можно считать, что равны -1 или 1).

        """
        pass

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает значение решающей функции.

        Parameters
        ----------
        X : np.ndarray
            Данные, для которых нужно посчитать значение решающей функции.

        Return
        ------
        np.ndarray
            Значение решающей функции для каждого элемента X
            (т.е. то число, от которого берем знак с целью узнать класс).

        """
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Классифицирует элементы X.

        Parameters
        ----------
        X : np.ndarray
            Данные, которые нужно классифицировать

        Return
        ------
        np.ndarray
            Метка класса для каждого элемента X.

        """
        return np.sign(self.decision_function(X))