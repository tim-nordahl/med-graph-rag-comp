from __future__ import annotations

from abc import ABC, abstractmethod

from src.schema import Passage, Question, RetrievalResult


class Retriever(ABC):
    name = "base"

    @abstractmethod
    def index(self, passages: list[Passage]) -> None: ...

    @abstractmethod
    def retrieve(self, questions: list[Question], k: int) -> list[RetrievalResult]: ...
