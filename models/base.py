"""统一模型接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def fit(self, features, labels):
        raise NotImplementedError

    @abstractmethod
    def predict(self, features):
        raise NotImplementedError
