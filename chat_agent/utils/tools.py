"""
Инструменты для ИИ-агента.

Модуль предоставляет функции для создания и конфигурации инструментов,
используемых агентом при взаимодействии с внешними источниками данных.
Включает поисковый инструмент через DuckDuckGo и ретривер для работы
с векторным хранилищем документов.

Functions
-------
get_search_tool
    Создаёт и возвращает инструмент поиска в интернете.
get_retrieved_tool_description
    Загружает описание для инструмента ретривера.
get_retriever_tool
    Создаёт и настраивает ретривер для поиска по документам.
"""
import os

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import create_retriever_tool

from .vectorstore import FixedSQLiteVec


def get_search_tool():
    """
    Возвращает инструмент поиска в интернете с помощью DuckDuckGo.

    Returns
    ----------
    DuckDuckGoSearchRun
        Настроенный инструмент поиска для использования агентом.
    """
    return DuckDuckGoSearchRun()


def get_retrieved_tool_description() -> str:
    """
    Загружает описание для инструмента ретривера из файла.

    Returns
    ----------
    str
        Описание инструмента ретривера.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "retrieved_tool_description.txt")
    with open(file_path, encoding="utf-8") as file:
        retrieved_tool_description = file.read()
    retrieved_tool_description = retrieved_tool_description.replace("\n", "")
    return retrieved_tool_description


def get_retriever_tool(
    connection, table_name: str, embedding_function, document_count: int = 3
):
    """
    Возвращает ретривер для поиска по документам из SQLite.

    Parameters
    ----------
    connection : object
        Активное подключение к базе данных SQLite.
    table_name : str
        Имя таблицы в базе данных, содержащей векторизованные документы.
    embedding_function : Сallable
       Модель для преобразования текста в векторные представления.
    document_count : int
        Количество наиболее релевантных документов для возврата при поиске,
        по умолчанию 3.

    Returns
    ----------
    Tool
        Настроенный инструмент LangChain для извлечения документов
        из векторного хранилища.
    """
    retrieved_tool_description = get_retrieved_tool_description()
    retriever = FixedSQLiteVec(
        table=table_name, connection=connection, embedding=embedding_function
    ).as_retriever(search_kwargs={"k": document_count})
    return create_retriever_tool(
        retriever=retriever,
        name="ConstructionDocumentsRetriever",
        description=retrieved_tool_description,
    )
