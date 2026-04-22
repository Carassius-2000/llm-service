import sqlite3

from fastapi import Request


def get_db(request: Request) -> sqlite3.Connection:
    """
    Извлекает соединение с базой данных из состояния FastAPI приложения.

    Parameters
    ----------
    request : `Request`
        Объект HTTP-запроса FastAPI.

    Returns
    -------
    `sqlite3.Connection`
        Соединение с SQLite, сохранённое в состоянии приложения.
    """
    return request.app.state.db_connection
