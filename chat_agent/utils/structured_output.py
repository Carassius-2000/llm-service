"""
Pydantic модель структурированного вывода для ответов агента.

Classes
-------
Response
    Модель Pydantic, описывающая схему стандартизированного ответа агента.
"""

from typing import Literal

from pydantic import BaseModel


class Response(BaseModel):
    """
    Схема для ответов агента.

    Гарантирует, что каждый вывод агента содержит явное обоснование,
    опциональную директиву вызова инструмента и финальный ответ для
    пользователя.

    Attributes
    ----------
    reasoning : str
        Внутренняя цепочка рассуждений агента или пошаговое обоснование,
        использованное для формирования финального ответа.
    tool : Literal["ConstructionDocumentsRetriever", "duckduckgo_search"] | None
        Идентификатор инструмента, который вызвал агент.
    answer : str
        Финальный ответ для пользователя, сгенерированный агентом.
    """

    reasoning: str
    tool: Literal["ConstructionDocumentsRetriever", "duckduckgo_search"] | None = None
    answer: str
