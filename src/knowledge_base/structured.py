import json
import re
from typing import Any, Callable

from .clients import LLMClient


class StructuredLLMError(RuntimeError):
    pass


def _extract_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    m = re.search(r"([\[{].*[\]}])", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    raise json.JSONDecodeError("no JSON in LLM output", text, 0)


def call_structured(
    llm: LLMClient,
    prompt: str,
    *,
    validate: Callable[[Any], None] | None = None,
    max_retries: int = 3,
) -> Any:
    last_err: Exception | None = None
    current_prompt = prompt
    for attempt in range(max_retries):
        response = llm.complete(current_prompt)
        try:
            data = _extract_json(response)
            if validate is not None:
                validate(data)
            return data
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
            last_err = e
            current_prompt = (
                prompt
                + f"\n\n# Предыдущая попытка {attempt + 1} не дала валидный JSON: {e}.\n"
                "В новой попытке ОБЯЗАТЕЛЬНО сохрани рассуждения по шагам, "
                "но в самом конце после последнего шага выведи ровно один "
                "валидный JSON-блок, строго соответствующий формату из «Описания задачи»."
            )
    raise StructuredLLMError(f"failed after {max_retries} attempts: {last_err}")
