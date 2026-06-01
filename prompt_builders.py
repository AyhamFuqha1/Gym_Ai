import json
from typing import Any, Dict, List


def _json_payload(payload: Dict[str, Any]) -> str:
    return json.dumps(payload or {}, ensure_ascii=False, indent=2, default=str)


def build_training_plan_prompt(user_payload: dict, exercise_context: str) -> List[Dict[str, str]]:
    system_prompt = (
        "You are FitMind's RAG training-plan planner. Build safe, practical gym plans "
        "using the retrieved exercise context. Use only retrieved exercise context unless "
        "it is explicitly impossible. Do not invent exercise IDs. Return JSON only, with "
        "no markdown and no explanatory text outside the JSON object. Include reasons for "
        "chosen exercises, include injury warnings when relevant, and keep the output "
        "compatible with the TrainingPlanRAGResponse schema."
    )
    user_prompt = (
        "Create a structured training plan from this user payload and retrieved context.\n\n"
        "Required JSON shape:\n"
        "{\n"
        '  "status": "success",\n'
        '  "plan_type": "training",\n'
        '  "user_id": "from user payload",\n'
        '  "summary": "short explanation",\n'
        '  "days": [\n'
        '    {"day": 1, "focus": "Chest", "exercises": [\n'
        '      {"exercise_id": 1, "name": "Exercise name", "sets": 3, "reps": "8-12", "rest_seconds": 90, "intensity": "moderate", "reason": "why selected", "source_id": 1}\n'
        "    ]}\n"
        "  ],\n"
        '  "injury_warnings": [],\n'
        '  "sources": [{"source_id": 1, "source_table": "exercises", "source_name": "Exercise name", "score": 0.9, "reason_used": "why this source was used"}]\n'
        "}\n\n"
        "User payload:\n"
        f"{_json_payload(user_payload)}\n\n"
        "Retrieved exercise context:\n"
        f"{exercise_context or 'No retrieved exercise context provided.'}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_nutrition_plan_prompt(user_payload: dict, food_context: str) -> List[Dict[str, str]]:
    system_prompt = (
        "You are FitMind's RAG nutrition-plan planner. Build safe, practical nutrition "
        "plans using the retrieved food context. Use only retrieved food context unless "
        "it is explicitly impossible. Do not invent food IDs. Respect allergies, medical "
        "constraints, dislikes, and preferences from the user payload. Return JSON only, "
        "with no markdown and no explanatory text outside the JSON object. Include reasons "
        "for chosen foods, include allergy warnings when relevant, and keep the output "
        "compatible with the NutritionPlanRAGResponse schema."
    )
    user_prompt = (
        "Create a structured nutrition plan from this user payload and retrieved context.\n\n"
        "Required JSON shape:\n"
        "{\n"
        '  "status": "success",\n'
        '  "plan_type": "nutrition",\n'
        '  "user_id": "from user payload",\n'
        '  "daily_calorie_target": 2200,\n'
        '  "macro_targets": {"protein_g": 150, "carbs_g": 220, "fat_g": 70},\n'
        '  "meals": [\n'
        '    {"meal": "breakfast", "foods": [\n'
        '      {"food_id": 1, "name": "Food name", "serving_size": "100g", "quantity": 1, "calories": 120, "protein": 20, "carbs": 4, "fat": 2, "reason": "why selected", "source_id": 1}\n'
        "    ]}\n"
        "  ],\n"
        '  "allergy_warnings": [],\n'
        '  "sources": [{"source_id": 1, "source_table": "foods", "source_name": "Food name", "score": 0.9, "reason_used": "why this source was used"}]\n'
        "}\n\n"
        "User payload:\n"
        f"{_json_payload(user_payload)}\n\n"
        "Retrieved food context:\n"
        f"{food_context or 'No retrieved food context provided.'}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
