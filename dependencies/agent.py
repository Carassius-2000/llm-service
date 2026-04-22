import sqlite3

from openai import APIConnectionError
from fastapi import Depends, HTTPException

from chat_agent.agent import ReActAgent
from dependencies.database import get_db
from logger import console_logger


def get_agent(connection: sqlite3.Connection = Depends(get_db)) -> ReActAgent:
    """
    Dependency-функция, возвращающая экземпляр ReActAgent с заданным соединением с БД.

    Parameters
    ----------
    connection : `sqlite3.Connection, default=Depends(get_db)`
        Соединение с SQLite, полученное через зависимость FastAPI.

    Returns
    -------
    `ReActAgent`
        Инициализированный агент для обработки запросов.

    Raises
    ------
    HTTPException
        - 404 Not Found: если отсутствуют необходимые файлы моделей (FileNotFoundError).
        - 503 Service Unavailable: если не удалось подключиться к серверу нейросети (APIConnectionError).
    """
    try:
        return ReActAgent(connection=connection)
    except FileNotFoundError as e:
        message_error = f"Отсутствуют файлы с моделями. {e}"
        console_logger.error(message_error)
        raise HTTPException(status_code=404, detail=message_error)
    except APIConnectionError as e:
        message_error = (
            f"Не удалось подключиться к серверу нейросети. "
            f"Проверьте доступность сервера и настройки подключения. Ошибка: {e}"
        )
        console_logger.error(message_error)
        raise HTTPException(status_code=503, detail=message_error)
