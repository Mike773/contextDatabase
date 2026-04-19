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
    blocks.append("# Начни с шага 1\nШаг 1.")
    return "\n\n".join(blocks)
