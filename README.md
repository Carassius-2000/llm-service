# Чатовый агент
## Описание проекта
LLM Service с агентным RAG. [Ссылка на dataset для БД](https://www.kaggle.com/datasets/kiraidk/documents-about-building-for-rag).

## Основные возможности
`ReAct` агент c tools:
- web_search через [DuckDuckGo](https://duckduckgo.com/).
- retrieved_tool через [SQLiteVec](https://github.com/asg017/sqlite-vec).

Есть механизм [Self RAG](https://arxiv.org/abs/2310.11511), сохранения контекста сообщений, `Structured Output` в формате:

```json
{
  "reason": "Запрос содержит маркер времени сейчас. Требуется актуальная информация, которой нет в статичной базе знаний.",
  "tool": "duckduckgo_search",
  "answer": "Сейчас погода в Казани 9 градусов по Цельсию.",
}
```

В system prompt использовалась техника `few-shot prompting`. Для получения embedding использовалась модель [Qwen3-Embedding 0.6B - Q8_0](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF) с размерностью 1024. Тестировалась работа агента на LLM [Qwen3.5 4B - Q4_K_M](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF).

## Технологический стек

- Python;
- LangGraph;
- LangChain;
- SQLiteVec;
- ddgs;
- FastAPI;
- Pydantic.

## Запуск сервиса
1. Клонируйте репозиторый с GitHub.
```bash 
git clone https://github.com/Carassius-2000/llm-service
```
2. Установите Docker.

3. Создайте файл `.env` с настройками для доступа к серверу с LLM и Embedding Model. Структура файла должна быть такой:
```bash
API_HOST_DOCKER=YOUR_HOST
API_KEY=YOUR_API_KEY
MODEL_NAME=YOUR_MODEL_NAME
EMBEDDING_NAME=YOUR_EMBEDDING_NAME
```

4. Соберите и запустите сервис.
```bash
docker compose up --build
```

5. Проверить работоспособность системы можно либо открыв в браузере [http://localhost:81/](http://localhost:81/), либо посмотрев состояние контейнеров. Должен быть запущен контейнер `api`.
```bash
docker compose ps -a
```

С документацией Web API можно ознакомиться двумя способами:
- Swagger/OpenAPI: [http://localhost:81//docs](http://localhost:81//docs).
- Redocly: [http://localhost:81//redoc](http://localhost:81//redoc).

Для того чтобы остановить работу контейнера выполните:
```bash
docker compose stop
```