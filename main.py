from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from chat_agent.agent import ReActAgent
from chat_agent.utils.structured_output import Response
from database import get_sqlite_connection
from dependencies.agent import get_agent
from logger import log_info


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


@app.post("/chat_message")
def get_answer(
    user_prompt: str,
    react_agent: ReActAgent = Depends(get_agent),
    background_tasks: BackgroundTasks = None,
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
    background_tasks.add_task(log_info, "Ответ от агента получен.")
    return response


@app.delete("/clear_history")
def clear_history(
    react_agent: ReActAgent = Depends(get_agent),
    background_tasks: BackgroundTasks = None,
) -> dict[str, str]:
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
    background_tasks.add_task(log_info, "История чата очищена.")
    return {"status": "history cleared"}


@app.get("/")
async def healthcheck() -> dict[str, str]:
    """
    Точка входа в API. Используется для проверки работоспособности сервера.

    Returns
    -------
    dict[str, str]
        JSON, который подтверждает, что сервер с API успешно запущен.
    """
    return {"message": "Сервер с API запущен."}
