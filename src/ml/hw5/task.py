import numpy as np
import copy
from typing import List
import torch
from torch import nn
import torch.nn.functional as F


# Task 1

class Module:
    """
    Абстрактный класс. Его менять не нужно. Он описывает общий интерфейс взаимодествия со слоями нейронной сети.
    """
    def forward(self, x):
        pass
    
    def backward(self, d):
        pass
        
    def update(self, alpha):
        pass
    
    
class Linear(Module):
    """
    Линейный полносвязный слой.
    """
    def __init__(self, in_features: int, out_features: int):
        """
        Parameters
        ----------
        in_features : int
            Размер входа.
        out_features : int 
            Размер выхода.
    
        Notes
        -----
        W и b инициализируются случайно.
        """
        self.in_features = in_features
        self.out_features = out_features

        self.W = np.random.uniform(0.0, 1.0, size=(in_features, out_features))
        self.b = np.random.uniform(0.0, 1.0, size=(out_features,))

        self._x_ndim2 = None
        self._x_was_vector = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает y = Wx + b.

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.
            То есть, либо x вектор с in_features элементов,
            либо матрица размерности (batch_size, in_features).
    
        Return
        ------
        y : np.ndarray
            Выход после слоя.
            Либо вектор с out_features элементами,
            либо матрица размерности (batch_size, out_features)
        """
        if x.ndim == 1:
            self._x_was_vector = True
            self._x_ndim2 = x.reshape(1, -1)
        elif x.ndim == 2:
            self._x_was_vector = False
            self._x_ndim2 = x

        y2 = self._x_ndim2 @ self.W + self.b
        return y2.reshape(-1) if self._x_was_vector else y2

    
    def backward(self, d: np.ndarray) -> np.ndarray:
        """
        Cчитает градиент при помощи обратного распространения ошибки.

        Parameters
        ----------
        d : np.ndarray
            Градиент.
        Return
        ------
        np.ndarray
            Новое значение градиента.
        """
        if d.ndim == 1:
            d2 = d.reshape(1, -1)
        elif d.ndim == 2:
            d2 = d

        # Градиенты параметров
        # dW: (in, batch)@(batch, out) = (in, out)
        self.dW = self._x_ndim2.T @ d2
        self.db = d2.sum(axis=0)

        dx2 = d2 @ self.W.T  # (batch, out)@(out, in) = (batch, in)
        return dx2.reshape(-1) if self._x_was_vector else dx2



        
    def update(self, alpha: float) -> None:
        """
        Обновляет W и b с заданной скоростью обучения.

        Parameters
        ----------
        alpha : float
            Скорость обучения.
        """
        self.W -= alpha * self.dW
        self.b -= alpha * self.db
    

class ReLU(Module):
    """
    Слой, соответствующий функции активации ReLU. Данная функция возвращает новый массив, в котором значения меньшие 0 заменены на 0.
    """
    def __init__(self):
        self._mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает y = max(0, x).

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.
    
        Return
        ------
        y : np.ndarray
            Выход после слоя (той же размерности, что и вход).

        """
        self._mask = (x > 0)
        return np.maximum(0, x)
        
    def backward(self, d) -> np.ndarray:
        """
        Cчитает градиент при помощи обратного распространения ошибки.

        Parameters
        ----------
        d : np.ndarray
            Градиент.
        Return
        ------
        np.ndarray
            Новое значение градиента.
        """
        return d * self._mask
    

# Task 2

class Softmax(Module):
    def __init__(self):
        self.p = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x2 = x.reshape(1, -1)
            x2 = x2 - np.max(x2, axis=1, keepdims=True)
            e = np.exp(x2)
            p2 = e / np.sum(e, axis=1, keepdims=True)
            self.p = p2
            return p2.reshape(-1)
        elif x.ndim == 2:
            x2 = x - np.max(x, axis=1, keepdims=True)
            e = np.exp(x2)
            self.p = e / np.sum(e, axis=1, keepdims=True)
            return self.p

    def backward(self, d: np.ndarray) -> np.ndarray:
        if d.ndim == 1:
            d2 = d.reshape(1, -1)
            p2 = self.p  # (1, C)
            # J^T v для softmax
            s = np.sum(d2 * p2, axis=1, keepdims=True)
            dx2 = p2 * (d2 - s)
            return dx2.reshape(-1)

        elif d.ndim == 2:
            p = self.p  # (B, C)
            s = np.sum(d * p, axis=1, keepdims=True)
            dx = p * (d - s)
            return dx


    def update(self, alpha: float) -> None:
        return


class MLPClassifier:
    def __init__(self, modules: List[Module], epochs: int = 40, alpha: float = 0.01, batch_size: int = 32):
        """
        Parameters
        ----------
        modules : List[Module]
            Cписок, состоящий из ранее реализованных модулей и 
            описывающий слои нейронной сети. 
            В конец необходимо добавить Softmax.
        epochs : int
            Количество эпох обучения.
        alpha : float
            Cкорость обучения.
        batch_size : int
            Размер батча, используемый в процессе обучения.
        """
        modules.append(Softmax())
        self.modules = modules
        self.epochs = epochs
        self.alpha = alpha
        self.batch_size = batch_size

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Обучает нейронную сеть заданное число эпох. 
        В каждой эпохе необходимо использовать cross-entropy loss для обучения, 
        а так же производить обновления не по одному элементу, а используя батчи (иначе обучение будет нестабильным и полученные результаты будут плохими.

        Parameters
        ----------
        X : np.ndarray
            Данные для обучения.
        y : np.ndarray
            Вектор меток классов для данных.
        """
        n = X.shape[0]
        eps = 1e-12
        n_classes = int(np.max(y)) + 1
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=np.int64).reshape(-1)

        for _ in range(self.epochs):
            perm = np.random.permutation(n)
            Xs = X[perm]
            ys = y[perm]

            for start in range(0, n, self.batch_size):
                xb = Xs[start:start + self.batch_size]
                yb = ys[start:start + self.batch_size]
                B = xb.shape[0]

                p = self._forward(xb)

                Y = np.zeros((B, n_classes), dtype=float)
                Y[np.arange(B), yb] = 1.0

                d = -(Y / (p + eps)) / B

                for m in reversed(self.modules):
                    d = m.backward(d)

                for m in self.modules:
                    m.update(self.alpha)


        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает вероятности классов для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Данные для предсказания.
        
        Return
        ------
        np.ndarray
            Предсказанные вероятности классов для всех элементов X.
            Размерность (X.shape[0], n_classes)
        
        """
        return self._forward(X)


    def predict(self, X) -> np.ndarray:
        """
        Предсказывает метки классов для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Данные для предсказания.

        Return
        ------
        np.ndarray
            Вектор предсказанных классов
        
        """
        p = self.predict_proba(X)
        return np.argmax(p, axis=1)


    def _forward(self, X: np.ndarray) -> np.ndarray:
        out = X
        for m in self.modules:
            out = m.forward(out)
        return out
# Task 3

classifier_moons = MLPClassifier(modules=[Linear(2, 32), ReLU(), Linear(32,2)], epochs=120, alpha=0.05, batch_size=32) # Нужно указать гиперпараметры
classifier_blobs = MLPClassifier(modules=[Linear(2, 32), ReLU(), Linear(32,3)], epochs=120, alpha=0.05, batch_size=32) # Нужно указать гиперпараметры


# Task 4

class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))


        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def load_model(self):
        """
        Используйте torch.load, чтобы загрузить обученную модель
        Учтите, что файлы решения находятся не в корне директории, поэтому необходимо использовать следующий путь:
        `__file__[:-7] + "model.pth"`, где "model.pth" - имя файла сохраненной модели `
        """
        state = torch.load(__file__[:-7] + "model.pth", map_location="cpu")
        self.load_state_dict(state)
        self.eval()
        return self

    def save_model(self):
        """
        Используйте torch.save, чтобы сохранить обученную модель
        """
        torch.save(self.state_dict(), __file__[:-7] + "model.pth")
        
def calculate_loss(X: torch.Tensor, y: torch.Tensor, model: TorchModel):
    """
    Cчитает cross-entropy.

    Parameters
    ----------
    X : torch.Tensor
        Данные для обучения.
    y : torch.Tensor
        Метки классов.
    model : Model
        Модель, которую будем обучать.

    """
    logits = model(X)
    y = y.long().reshape(-1)
    return torch.nn.functional.cross_entropy(logits, y)
