def build_prompt(
    task: str,
    title: str,
    text: str,
    steps: list[str],
    *,
    context: str | None = None,
) -> str:
    numbered = "\n".join(f"Шаг {i + 1}. {s}" for i, s in enumerate(steps))
    blocks: list[str] = []
    if context and context.strip():
        blocks.append(f"# Контекст\n{context.strip()}")
    blocks.append(f"# Описание задачи\n{task}")
    blocks.append(f"# Документ\nНазвание: {title}\n\n{text}")
    blocks.append(f"# Структура рассуждения по шагам\n{numbered}")
    blocks.append(
        "# Начни с шага 1\n"
        f"Выполни ВСЕ {len(steps)} шагов последовательно, каждый начинай "
        "с новой строки вида «Шаг N.». Не пропускай шаги и не сокращай "
        "рассуждения. Только после последнего шага выдай финальный "
        "ответ, указанный в «Описании задачи».\n\n"
        "Шаг 1."
    )
    return "\n\n".join(blocks)
