import sqlite3

import sqlite_vec


def get_sqlite_connection() -> sqlite3.Connection:
    """
    Создаёт и возвращает соединение с SQLiteVec.

    Returns
    -------
    `sqlite3.Connection`
        Активное соединение с SQLite, готовое к выполнению запросов.
    """
    connection = sqlite3.connect("my_vectors.db", check_same_thread=False)
    connection.enable_load_extension(True)
    sqlite_vec.load(connection)
    connection.enable_load_extension(False)
    return connection
