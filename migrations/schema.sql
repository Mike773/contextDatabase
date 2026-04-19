-- Актуальная схема базы знаний агента-аналитика.
-- Один файл = текущая версия. Индексы не включены (добавляются отдельно).
-- Требуется расширение pgvector.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS rag_v2;
SET search_path TO rag_v2, public;

-- 1. Направления (верхний уровень изоляции)
CREATE TABLE IF NOT EXISTS rag_v2.directions (
    id                    BIGSERIAL PRIMARY KEY,
    short_name            TEXT NOT NULL,
    full_name             TEXT NOT NULL,
    general_info          TEXT,
    abbreviations         JSONB NOT NULL DEFAULT '{}'::jsonb,
    short_name_embedding  VECTOR(2560),
    full_name_embedding   VECTOR(2560)
);

-- 2. Сырые документы, привязанные к направлению.
-- analysis_plan JSONB: {"criteria": {<key>: bool, ...}, "algorithms_to_run": [<key>, ...], "reasoning": "<...>"}
CREATE TABLE IF NOT EXISTS rag_v2.documents (
    id                 BIGSERIAL PRIMARY KEY,
    direction_id       BIGINT NOT NULL REFERENCES rag_v2.directions(id) ON DELETE CASCADE,
    title              TEXT NOT NULL,
    text               TEXT NOT NULL,
    should_chunk       BOOLEAN NOT NULL DEFAULT TRUE,
    chunk_separator    TEXT,
    summary            TEXT,
    unclear_items      JSONB NOT NULL DEFAULT '{}'::jsonb,
    analysis_plan      JSONB NOT NULL DEFAULT '{}'::jsonb,
    title_embedding    VECTOR(2560),
    summary_embedding  VECTOR(2560)
);

-- 3. Чанки документов
CREATE TABLE IF NOT EXISTS rag_v2.chunks (
    id              BIGSERIAL PRIMARY KEY,
    document_id     BIGINT NOT NULL REFERENCES rag_v2.documents(id)  ON DELETE CASCADE,
    direction_id    BIGINT NOT NULL REFERENCES rag_v2.directions(id) ON DELETE CASCADE,
    text            TEXT NOT NULL,
    text_embedding  VECTOR(2560)
);

-- 4. Термины / наименования / процессы, выделенные из документов.
-- scope: 'organization' — про организацию в целом; 'direction' — про конкретное направление.
CREATE TABLE IF NOT EXISTS rag_v2.terms (
    id                          BIGSERIAL PRIMARY KEY,
    direction_id                BIGINT NOT NULL REFERENCES rag_v2.directions(id) ON DELETE CASCADE,
    scope                       TEXT NOT NULL DEFAULT 'direction',
    name                        TEXT NOT NULL,
    short_description           TEXT,
    detailed_description        TEXT,
    quotes                      JSONB NOT NULL DEFAULT '[]'::jsonb,
    name_embedding              VECTOR(2560),
    short_description_embedding VECTOR(2560)
);

-- 5. Роли (detailed_description включает связи между ролями)
CREATE TABLE IF NOT EXISTS rag_v2.roles (
    id                          BIGSERIAL PRIMARY KEY,
    direction_id                BIGINT NOT NULL REFERENCES rag_v2.directions(id) ON DELETE CASCADE,
    name                        TEXT NOT NULL,
    short_description           TEXT,
    detailed_description        TEXT,
    quotes                      JSONB NOT NULL DEFAULT '[]'::jsonb,
    name_embedding              VECTOR(2560),
    short_description_embedding VECTOR(2560)
);

-- 6. Утверждения из документов.
-- scope: 'organization' — про организацию в целом; 'direction' — про конкретное направление.
CREATE TABLE IF NOT EXISTS rag_v2.claims (
    id                          BIGSERIAL PRIMARY KEY,
    direction_id                BIGINT NOT NULL REFERENCES rag_v2.directions(id) ON DELETE CASCADE,
    scope                       TEXT NOT NULL DEFAULT 'direction',
    short_description           TEXT NOT NULL,
    detailed_description        TEXT NOT NULL,
    document_ids                BIGINT[] NOT NULL DEFAULT '{}',
    short_description_embedding VECTOR(2560)
);

-- 7. Противоречия между документами
CREATE TABLE IF NOT EXISTS rag_v2.contradictions (
    id                          BIGSERIAL PRIMARY KEY,
    direction_id                BIGINT NOT NULL REFERENCES rag_v2.directions(id) ON DELETE CASCADE,
    short_description           TEXT NOT NULL,
    detailed_description        TEXT NOT NULL,
    quotes                      JSONB NOT NULL DEFAULT '[]'::jsonb,
    document_ids                BIGINT[] NOT NULL DEFAULT '{}',
    short_description_embedding VECTOR(2560)
);

-- 8. Метрики / KPI / драйверы / показатели
CREATE TABLE IF NOT EXISTS rag_v2.metrics (
    id                          BIGSERIAL PRIMARY KEY,
    direction_id                BIGINT NOT NULL REFERENCES rag_v2.directions(id) ON DELETE CASCADE,
    name                        TEXT NOT NULL,
    short_description           TEXT,
    detailed_description        TEXT,
    connections_description     TEXT,
    document_ids                BIGINT[] NOT NULL DEFAULT '{}',
    role_ids                    BIGINT[] NOT NULL DEFAULT '{}',
    name_embedding              VECTOR(2560),
    short_description_embedding VECTOR(2560)
);

-- 9. Алгоритмы
CREATE TABLE IF NOT EXISTS rag_v2.algorithms (
    id                          BIGSERIAL PRIMARY KEY,
    direction_id                BIGINT NOT NULL REFERENCES rag_v2.directions(id) ON DELETE CASCADE,
    name                        TEXT NOT NULL,
    short_description           TEXT,
    detailed_description        TEXT NOT NULL,
    quotes                      JSONB NOT NULL DEFAULT '[]'::jsonb,
    document_ids                BIGINT[] NOT NULL DEFAULT '{}',
    metric_ids                  BIGINT[] NOT NULL DEFAULT '{}',
    role_ids                    BIGINT[] NOT NULL DEFAULT '{}',
    name_embedding              VECTOR(2560),
    short_description_embedding VECTOR(2560)
);

-- 10. Извлечённые из документов сущности — буфер перед загрузкой в основные таблицы.
-- Темповая таблица: без FK на directions/documents, чистится на стороне приложения.
CREATE TABLE IF NOT EXISTS rag_v2.extractions (
    id            BIGSERIAL PRIMARY KEY,
    direction_id  BIGINT NOT NULL,
    document_id   BIGINT NOT NULL,
    entity_type   TEXT NOT NULL,
    name          TEXT,
    description   TEXT,
    quote         TEXT,
    status        TEXT NOT NULL DEFAULT 'pending'
);
