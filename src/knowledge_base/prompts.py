def build_prompt(task: str, title: str, text: str, steps: list[str]) -> str:
    numbered = "\n".join(f"Шаг {i + 1}. {s}" for i, s in enumerate(steps))
    return (
        "# Описание задачи\n"
        f"{task}\n\n"
        "# Документ\n"
        f"Название: {title}\n\n"
        f"{text}\n\n"
        "# Структура рассуждения по шагам\n"
        f"{numbered}\n\n"
        "# Начни с шага 1\n"
        "Шаг 1."
    )
