-- Актуальная схема базы знаний агента-аналитика.
-- Один файл = текущая версия. Индексы не включены (добавляются отдельно).
-- Требуется расширение pgvector.

CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Направления (верхний уровень изоляции)
CREATE TABLE IF NOT EXISTS directions (
    id                    BIGSERIAL PRIMARY KEY,
    short_name            TEXT NOT NULL,
    full_name             TEXT NOT NULL,
    abbreviations         JSONB NOT NULL DEFAULT '{}'::jsonb,
    short_name_embedding  VECTOR(2560),
    full_name_embedding   VECTOR(2560)
);

-- 2. Сырые документы, привязанные к направлению
CREATE TABLE IF NOT EXISTS documents (
    id                 BIGSERIAL PRIMARY KEY,
    direction_id       BIGINT NOT NULL REFERENCES directions(id) ON DELETE CASCADE,
    title              TEXT NOT NULL,
    text               TEXT NOT NULL,
    should_chunk       BOOLEAN NOT NULL DEFAULT TRUE,
    chunk_separator    TEXT,
    summary            TEXT,
    unclear_items      JSONB NOT NULL DEFAULT '{}'::jsonb,
    title_embedding    VECTOR(2560),
    summary_embedding  VECTOR(2560)
);

-- 3. Чанки документов
CREATE TABLE IF NOT EXISTS chunks (
    id              BIGSERIAL PRIMARY KEY,
    document_id     BIGINT NOT NULL REFERENCES documents(id)  ON DELETE CASCADE,
    direction_id    BIGINT NOT NULL REFERENCES directions(id) ON DELETE CASCADE,
    text            TEXT NOT NULL,
    text_embedding  VECTOR(2560)
);

-- 4. Термины / наименования / процессы, выделенные из документов
CREATE TABLE IF NOT EXISTS terms (
    id                          BIGSERIAL PRIMARY KEY,
    direction_id                BIGINT NOT NULL REFERENCES directions(id) ON DELETE CASCADE,
    name                        TEXT NOT NULL,
    short_description           TEXT,
    detailed_description        TEXT,
    name_embedding              VECTOR(2560),
    short_description_embedding VECTOR(2560)
);

-- 5. Роли (detailed_description включает связи между ролями)
CREATE TABLE IF NOT EXISTS roles (
    id                          BIGSERIAL PRIMARY KEY,
    direction_id                BIGINT NOT NULL REFERENCES directions(id) ON DELETE CASCADE,
    name                        TEXT NOT NULL,
    short_description           TEXT,
    detailed_description        TEXT,
    name_embedding              VECTOR(2560),
    short_description_embedding VECTOR(2560)
);

-- 6. Утверждения из документов
CREATE TABLE IF NOT EXISTS claims (
    id                          BIGSERIAL PRIMARY KEY,
    direction_id                BIGINT NOT NULL REFERENCES directions(id) ON DELETE CASCADE,
    short_description           TEXT NOT NULL,
    detailed_description        TEXT NOT NULL,
    document_ids                BIGINT[] NOT NULL DEFAULT '{}',
    short_description_embedding VECTOR(2560)
);

-- 7. Противоречия между документами
CREATE TABLE IF NOT EXISTS contradictions (
    id                          BIGSERIAL PRIMARY KEY,
    direction_id                BIGINT NOT NULL REFERENCES directions(id) ON DELETE CASCADE,
    short_description           TEXT NOT NULL,
    detailed_description        TEXT NOT NULL,
    quotes                      JSONB NOT NULL DEFAULT '[]'::jsonb,
    document_ids                BIGINT[] NOT NULL DEFAULT '{}',
    short_description_embedding VECTOR(2560)
);

-- 8. Метрики / KPI / драйверы / показатели
CREATE TABLE IF NOT EXISTS metrics (
    id                          BIGSERIAL PRIMARY KEY,
    direction_id                BIGINT NOT NULL REFERENCES directions(id) ON DELETE CASCADE,
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
CREATE TABLE IF NOT EXISTS algorithms (
    id                          BIGSERIAL PRIMARY KEY,
    direction_id                BIGINT NOT NULL REFERENCES directions(id) ON DELETE CASCADE,
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
