import json
import re
from typing import Any, Dict

from rag_schemas import NutritionPlanRAGResponse, TrainingPlanRAGResponse


def _ensure_json_object(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("LLM response JSON must be an object")
    return value


def extract_json_object(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("No response text provided")

    stripped = text.strip()

    try:
        return _ensure_json_object(json.loads(stripped))
    except json.JSONDecodeError:
        pass

    code_blocks = re.findall(r"```(?:json)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL)
    for block in code_blocks:
        try:
            return _ensure_json_object(json.loads(block.strip()))
        except json.JSONDecodeError:
            continue

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", stripped):
        candidate = stripped[match.start():]
        try:
            obj, _ = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        return _ensure_json_object(obj)

    raise ValueError("No valid JSON object found in LLM response")


def parse_training_plan_response(text: str) -> TrainingPlanRAGResponse:
    data = extract_json_object(text)
    return TrainingPlanRAGResponse.model_validate(data)


def parse_nutrition_plan_response(text: str) -> NutritionPlanRAGResponse:
    data = extract_json_object(text)
    return NutritionPlanRAGResponse.model_validate(data)
