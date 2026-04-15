"""
Реализация ReAct-агента с поддержкой инструментов, памяти и PII-фильтрации.

Модуль содержит класс ReActAgent, который инкапсулирует логику создания
агента LangChain, подключения инструментов поиска и RAG, а также управление
историей диалогов через SQLite-чекпоинтер.
"""

import os
import re

from dotenv import load_dotenv
from IPython.display import Image, display
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver

from chat_agent.utils.structured_output import Response
from chat_agent.utils.tools import get_retriever_tool, get_search_tool


class ReActAgent:
    """
    ReAct агент для взаимодействия с внешними инструментами и векторным хранилищем.

    Инициализирует LLM, эмбеддинги, инструменты поиска и RAG-ретривер.
    Поддерживает сохранение истории диалогов, маскировку персональных данных
    и парсинг структурированных ответов.

    Attributes
    --------
    _middlewares : list[PIIMiddleware]
        Cписок middleware для маскировки email, карт, IP, MAC и URL
        во входных данных, ответах и результатах вызова инструментов.
    agent : AgentExecutor
        Агент LangGraph с привязанными инструментами и памятью.
    """

    _middlewares = [
        PIIMiddleware(
            "email",
            strategy="mask",
            apply_to_input=True,
            apply_to_output=True,
            apply_to_tool_results=True,
        ),
        PIIMiddleware(
            "credit_card",
            strategy="mask",
            apply_to_input=True,
            apply_to_output=True,
            apply_to_tool_results=True,
        ),
        PIIMiddleware(
            "ip",
            strategy="mask",
            apply_to_input=True,
            apply_to_output=True,
            apply_to_tool_results=True,
        ),
        PIIMiddleware(
            "mac_address",
            strategy="mask",
            apply_to_input=True,
            apply_to_output=True,
            apply_to_tool_results=True,
        ),
        PIIMiddleware(
            "url",
            strategy="mask",
            apply_to_input=True,
            apply_to_output=True,
            apply_to_tool_results=True,
        ),
    ]

    def __init__(
        self,
        connection,
        table_name: str = "my_vectors",
        document_count: int = 3,
        thread_id: str = "document-session",
    ):
        """
        Инициализирует агент с LLM, эмбеддингами и инструментами.

        Parameters
        ---------
        connection : sqlite3.Connection
            Подключение к SQLite-базе для чекпоинтера и векторного хранилища.
        table_name : str
            Имя таблицы с векторизованными документами, по умолчанию my_vectors.
        document_count : int
            Количество возвращаемых документов при RAG-поиске, по умолчанию 3.
        thread_id : str
            Идентификатор сессии диалога для изоляции памяти, по умолчанию document-session.
        """
        llm = ChatOpenAI(
            base_url=os.getenv("API_HOST_DOCKER"),
            api_key=os.getenv("API_KEY"),
            model=os.getenv("MODEL_NAME"),
        )
        embedding_function = OpenAIEmbeddings(
            base_url=os.getenv("API_HOST_DOCKER"),
            api_key=os.getenv("API_KEY"),
            model=os.getenv("EMBEDDING_NAME"),
            check_embedding_ctx_length=False,
        )
        tools = [
            get_search_tool(),
            get_retriever_tool(
                connection=connection,
                table_name=table_name,
                embedding_function=embedding_function,
                document_count=document_count,
            ),
        ]
        self.__checkpointer = SqliteSaver(connection)
        system_prompt = self.__load_system_prompt()
        self.agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            middleware=self._middlewares,
            checkpointer=self.__checkpointer,
        )
        self.__config = {"configurable": {"thread_id": thread_id}}

    def __load_system_prompt(self) -> str:
        """
        Загружает системный промпт из файла system_prompt.txt.

        Returns
        ----------
        str
            Системный промпт для агента.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "system_prompt.txt")
        with open(file_path, encoding="utf-8") as file:
            return file.read().replace("\n", " ").strip()

    @staticmethod
    def __get_structured_response(model_response: str) -> Response:
        """
        Парсит ответ модели в структурированный формат Response.

        Parameters
        ---------
        model_response : str
            Сырой текст ответа LLM.

        Returns
        -------
        Response
            Pydantic модель для ответа агента.
        """
        pattern = r"\[REASONING\]:\s*(.*?)\s*\[TOOL\]:\s*(.*?)\s*\[ANSWER\]:\s*(.*)"
        match = re.search(pattern, model_response, re.DOTALL)
        reasoning = match.group(1).strip()
        tool = None if match.group(2).strip() == "None" else match.group(2).strip()
        answer = match.group(3).strip()
        return Response(reasoning=reasoning, tool=tool, answer=answer)

    def invoke(self, user_prompt: str) -> Response:
        """
        Вызов агента с пользовательским запросом и возвращает структурированный ответ.

        Parameters
        ---------
        user_prompt : str
            Пользовательский промпт.

        Returns
        ----------
        Response
            Структурированный ответ агента.
        """
        response = self.agent.invoke(
            {"messages": [HumanMessage(content=user_prompt.lower())]},
            config=self.__config,
        )
        last_message = response["messages"][-1].content
        return self.__get_structured_response(last_message)

    def clear_history(self) -> None:
        """
        Очищает историю диалога для текущей сессии.
        """
        thread_id = self.__config["configurable"]["thread_id"]
        self.__checkpointer.delete_thread(thread_id)

    def display_graph(self):
        """
        Визуализирует граф выполнения агента.
        """
        display(Image(self.agent.get_graph().draw_png()))

    def draw_ascii(self):
        """
        Возвращает текстовое представление цикла агента.

        Возвращает
        ----------
        str
            ASCII-диаграмма графа LangGraph.
        """
        return self.agent.get_graph().draw_ascii()
