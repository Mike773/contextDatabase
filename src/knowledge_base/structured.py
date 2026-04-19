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
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Walk the text left-to-right; at each '{' or '[' try raw_decode.
    # On success, advance past the decoded region so inner structures inside an
    # already-consumed JSON are not revisited. Keep the LAST successful decode —
    # models typically emit the real final JSON after reasoning prose that may
    # contain pseudo-lists like "[foo, bar]" which raw_decode will reject.
    dec = json.JSONDecoder()
    last_obj: Any = None
    i = 0
    n = len(text)
    while i < n:
        if text[i] in "{[":
            try:
                obj, end = dec.raw_decode(text[i:])
                last_obj = obj
                i += end
                continue
            except json.JSONDecodeError:
                pass
        i += 1
    if last_obj is not None:
        return last_obj
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
