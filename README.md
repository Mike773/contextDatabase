# contextDatabase

База знаний для агента-аналитика: извлечение структурированной информации из документов направлений (роли, метрики, алгоритмы, утверждения, термины) в PostgreSQL с pgvector.

## Требования

- PostgreSQL 14+ с расширением `pgvector`
- Python 3.10+
- Инжектируемые клиенты LLM и embedding (пакет не включает конкретные реализации)

## Установка

### 1. База данных

```bash
docker run -d --name kb-postgres -p 5433:5432 \
  -e POSTGRES_PASSWORD=<пароль> \
  -e POSTGRES_DB=knowledge \
  pgvector/pgvector:pg16
```

### 2. Схема

Единственный файл [migrations/schema.sql](migrations/schema.sql) описывает актуальную версию всей схемы в `rag_v2`. Перед применением проверь `VECTOR(2560)` — должна совпадать с размерностью твоей embedding-модели; если нет — замени 2560 на нужное число.

```bash
psql postgresql://postgres:<пароль>@localhost:5433/knowledge \
  -f migrations/schema.sql
```

### 3. Python-пакет

```bash
pip install -e .
```

Зависимости: `psycopg2-binary`, `pgvector`.

## Конфигурация клиентов

Пакет принимает LLM и embedding как реализации `Protocol`-ов. Опиши свои:

```python
from knowledge_base import KnowledgeBase, LLMClient, EmbeddingClient


class MyLLM:
    def complete(self, prompt: str) -> str:
        # вызов локальной или внешней LLM
        ...


class MyEmbedding:
    def embed(self, text: str) -> list[float]:
        # ДОЛЖЕН возвращать вектор размерности, равной VECTOR(N) из schema.sql
        ...


kb = KnowledgeBase(
    dsn="postgresql://postgres:<пароль>@localhost:5433/knowledge",
    llm=MyLLM(),
    embedding=MyEmbedding(),
)
```

## Загрузка и обработка документа

### 1. Создать направление (direction)

Направление — верхний уровень изоляции. Все документы, термины, роли и метрики относятся к одному направлению.

```python
import psycopg2.extras

with kb.db.cursor() as cur:
    cur.execute(
        "INSERT INTO rag_v2.directions "
        "(short_name, full_name, general_info, abbreviations) "
        "VALUES (%s, %s, %s, %s) RETURNING id",
        (
            "ФинАн",
            "Финансовая аналитика",
            "Анализ денежных потоков, планирование и контроль KPI.",
            psycopg2.extras.Json({
                "KPI": "Ключевой показатель эффективности",
                "НО": "Начальник отдела",
                "ФД": "Финансовый директор",
            }),
        ),
    )
    direction_id = cur.fetchone()["id"]
kb.db.commit()
```

Чем полнее `abbreviations`, тем лучше качество анализа — LLM строго запрещено выдумывать расшифровки, он опирается на этот словарь.

### 2. Загрузить документ

```python
with kb.db.cursor() as cur:
    cur.execute(
        "INSERT INTO rag_v2.documents (direction_id, title, text) "
        "VALUES (%s, %s, %s) RETURNING id",
        (direction_id, "Регламент расчёта KPI для отдела продаж",
         "<полный текст документа>"),
    )
    document_id = cur.fetchone()["id"]
kb.db.commit()
```

### 3. Прогнать анализаторы

Прямой порядок, все 7:

```python
for name in (
    "summary",        # краткое саммари документа
    "analysis_plan",  # какие из критериев применимы
    "organization",   # инфа о компании в целом
    "direction",      # инфа о направлении
    "roles",          # роли + процессы/церемонии для них
    "metrics",        # метрики и показатели
    "algorithms",     # алгоритмы и правила интерпретации
):
    kb.run(name, document_id)
```

Условный порядок по решению планировщика (экономит вызовы):

```python
kb.run("summary", document_id)
kb.run("analysis_plan", document_id)

plan = kb.db.fetch_document(document_id)["analysis_plan"]
for name in plan["algorithms_to_run"]:
    kb.run(name, document_id)
```

Каждый анализатор — идемпотентная перезапись целевых полей документа; для сущностных таблиц (`terms`, `claims`) действует дедуп внутри анализатора. Безопасно перезапускать.

### Что куда пишется

| Анализатор | Пишет в | Что |
|---|---|---|
| `summary` | `documents.summary` + `summary_embedding` | Тема + саммари + ключевые темы, склеенные и эмбеддированные |
| `analysis_plan` | `documents.analysis_plan` (JSONB) | `{criteria: {org,dir,roles,algorithms,metrics: bool}, algorithms_to_run: [...], reasoning}` |
| `organization` | `terms` (scope=organization) + `claims` (scope=organization) | Термины и утверждения о компании в целом |
| `direction` | `terms` (scope=direction) + `claims` (scope=direction) | Термины и утверждения о конкретном направлении |
| `roles` | `extractions` (entity_type=role) + `claims` (scope=role) | Роли с `alternative_names`, а также role_claims — процессы/церемонии ролей с `duration`/`periodicity`/`conditions` в начале `detailed_description` |
| `metrics` | `extractions` (entity_type=metric) | Метрики с описанием или контекстом использования |
| `algorithms` | `extractions` (entity_type=algorithm) | Алгоритмы, процессы, правила интерпретации метрик |

### Промоушен

Записи в `extractions` — это буфер. Дедуп и загрузка в типизированные таблицы (`roles`, `metrics`, `algorithms`) выполняются отдельным процессом, не входящим в этот пакет.

## CLI

Шаблон [run.py](run.py) в корне проекта — собственный entry point. Подставь свои `MyLLM`/`MyEmbedding` и DSN:

```bash
python run.py <analyzer_name> <document_id>
```

Доступные имена анализаторов: `summary`, `analysis_plan`, `organization`, `direction`, `roles`, `metrics`, `algorithms`.

## Устройство промптов

Все промпты в анализаторах построены по единому шаблону `knowledge_base.prompts.build_prompt`:

```
# Контекст
<направление, аббревиатуры, саммари документа, уже известные сущности>

# Описание задачи
<что нужно извлечь, формат финального JSON, жёсткое правило: не выдумывать аббревиатуры>

# Документ
Название: <title>

<text>

# Структура рассуждения по шагам
Шаг 1. ...
Шаг 2. ...
...

# Начни с шага 1
Выполни ВСЕ N шагов последовательно...

Шаг 1.
```

LLM обязана выписать рассуждения по шагам до финального JSON. `knowledge_base.structured.call_structured` парсит JSON из свободного текста и ретраит при ошибке, сохраняя рассуждения в промпте для повторной попытки.

## Эмбеддинги и аббревиатуры

Перед каждым `embedding.embed(text)` пакет подставляет полные расшифровки аббревиатур из `direction.abbreviations` вместо самих сокращений. Модель семантически не понимает «KPI», но понимает «Ключевой показатель эффективности» — так поиск по смыслу работает корректно. Сам LLM-промпт по-прежнему получает аббревиатуры в исходном виде.
