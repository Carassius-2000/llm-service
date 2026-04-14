import sqlite3
from contextlib import asynccontextmanager

from openai import APIConnectionError
import sqlite_vec
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from chat_agent.agent import ReActAgent
from chat_agent.utils.structured_output import Response
from logger import console_logger


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управляет жизненным циклом FastAPI приложения: создаёт и закрывает соединение с БД.

    Parameters
    ----------
    app : FastAPI
        Экземпляр FastAPI приложения, в состояние которого сохраняется соединение.

    Yields
    ------
    None
        Управление передаётся приложению. После завершения работы приложения
        выполняется закрытие соединения.
    """
    connection = get_sqlite_connection()
    app.state.db_connection = connection
    yield
    connection.close()


app = FastAPI(
    title="LLM Web Service for chating",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui() -> HTMLResponse:
    """
    Возвращает HTML-ответ с настроенной страницей Swagger UI,
    которая предоставляет интерфейс для взаимодействия с API.

    Используется для настройки кастомного Swagger UI.

    Returns
    -------
    HTMLResponse
        HTML-страница с интерфейсом Swagger UI.
    """
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )


@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
async def swagger_ui_redirect() -> HTMLResponse:
    """
    Возвращает HTML-ответ, используемый при аутентификации OAuth2
    через Swagger UI.

    Returns
    -------
    HTMLResponse
        HTML-ответ, необходимый для завершения OAuth2 flow в Swagger UI.
    """
    return get_swagger_ui_oauth2_redirect_html()


@app.get("/redoc", include_in_schema=False)
async def custom_redoc() -> HTMLResponse:
    """
    Возвращает HTML-ответ с интерфейсом ReDoc, который предоставляет
    альтернативный способ просмотра документации OpenAPI.
    Используется для настройки кастомного интерфейса ReDoc.

    Returns
    -------
    HTMLResponse
        HTML-страница с интерфейсом ReDoc.
    """
    return get_redoc_html(
        openapi_url="/openapi.json",
        title="ReDoc",
        redoc_js_url="/static/redoc.standalone.js",
    )


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


@app.post("/chat_message")
def get_answer(
    user_prompt: str, react_agent: ReActAgent = Depends(get_agent)
) -> Response:
    """
    Отправляет пользовательский запрос агенту и возвращает структурированный ответ.

    Parameters
    ----------
    user_prompt : `str`
        Текст сообщения пользователя.
        
    react_agent : `ReActAgent, default=Depends(get_agent)`
        Экземпляр агента, полученный через dependency injection.

    Returns
    -------
    `Response`
        Структурированный ответ агента (обычно содержит текст ответа и метаданные).
    """
    response = react_agent.invoke(user_prompt)
    console_logger.info("Ответ от агента получен.")
    return response


@app.delete("/clear_history")
def clear_history(react_agent: ReActAgent = Depends(get_agent)) -> dict[str, str]:
    """
    Очищает историю текущего диалога агента.

    Parameters
    ----------
    react_agent : `ReActAgent, default=Depends(get_agent)`
        Экземпляр агента, полученный через dependency injection.

    Returns
    -------
    `dict[str, str]`
        Словарь с ключом "status" и значением "history cleared",
        подтверждающий успешную очистку истории.
    """
    react_agent.clear_history()
    console_logger.info("История чата очищена.")
    return {"status": "history cleared"}


@app.get("/")
async def root() -> dict[str, str]:
    """
    Точка входа в API. Используется для проверки работоспособности сервера.

    Returns
    -------
    dict[str, str]
        JSON, который подтверждает, что сервер с API успешно запущен.
    """
    return {"message": "Сервер с API запущен."}
