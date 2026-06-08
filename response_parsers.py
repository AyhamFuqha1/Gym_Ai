import json
import re
from typing import Any, Dict

from rag_schemas import (
    ChatAssistantRAGResponse,
    ModifiedNutritionPlanRAGResponse,
    ModifiedTrainingPlanRAGResponse,
    NutritionPlanRAGResponse,
    TrainingPlanRAGResponse,
)


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


def parse_chat_assistant_response(text: str) -> ChatAssistantRAGResponse:
    data = extract_json_object(text)
    return ChatAssistantRAGResponse.model_validate(data)


def normalize_training_exercise_source_ids(data: Dict[str, Any], plan_key: str = "days") -> Dict[str, Any]:
    if plan_key == "modified_plan":
        plan = data.get("modified_plan", {})
        days = plan.get("schedule", []) if isinstance(plan, dict) else []
    else:
        days = data.get(plan_key, [])

    if not isinstance(days, list):
        return data

    for day in days:
        if not isinstance(day, dict):
            continue
        exercises = day.get("exercises", [])
        if not isinstance(exercises, list):
            continue
        for exercise in exercises:
            if not isinstance(exercise, dict):
                continue
            if exercise.get("source_id") in [None, ""] and exercise.get("exercise_id") not in [None, ""]:
                exercise["source_id"] = exercise["exercise_id"]

    return data


def parse_modified_training_plan_response(text: str) -> ModifiedTrainingPlanRAGResponse:
    data = extract_json_object(text)
    data = normalize_training_exercise_source_ids(data, plan_key="modified_plan")
    return ModifiedTrainingPlanRAGResponse.model_validate(data)


def normalize_nutrition_food_source_ids(data: Dict[str, Any], plan_key: str = "meals") -> Dict[str, Any]:
    if plan_key == "modified_plan":
        plan = data.get("modified_plan", {})
        if isinstance(plan, dict):
            meals = plan.get("daily_meals", [])
            if not meals and isinstance(plan.get("meals"), list):
                meals = plan.get("meals", [])
                plan["daily_meals"] = meals
        else:
            meals = []
    else:
        meals = data.get("meals", [])

    if not isinstance(meals, list):
        return data

    for meal in meals:
        if not isinstance(meal, dict):
            continue
        foods = meal.get("foods", [])
        if not foods and isinstance(meal.get("items"), list):
            foods = meal.get("items", [])
            meal["foods"] = foods
        if not isinstance(foods, list):
            continue
        for food in foods:
            if not isinstance(food, dict):
                continue
            if food.get("source_id") in [None, ""] and food.get("food_id") not in [None, ""]:
                food["source_id"] = food["food_id"]

    return data


def parse_nutrition_plan_response(text: str) -> NutritionPlanRAGResponse:
    data = extract_json_object(text)
    data = normalize_nutrition_food_source_ids(data)
    return NutritionPlanRAGResponse.model_validate(data)


def parse_modified_nutrition_plan_response(text: str) -> ModifiedNutritionPlanRAGResponse:
    data = extract_json_object(text)
    data = normalize_nutrition_food_source_ids(data, plan_key="modified_plan")
    return ModifiedNutritionPlanRAGResponse.model_validate(data)
