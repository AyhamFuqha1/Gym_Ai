import json
from typing import Any, Dict, List


def _json_payload(payload: Dict[str, Any]) -> str:
    return json.dumps(payload or {}, ensure_ascii=False, indent=2, default=str)


def build_training_plan_prompt(user_payload: dict, exercise_context: str) -> List[Dict[str, str]]:
    system_prompt = (
        "You are FitMind's RAG training-plan planner. Build safe, practical gym plans "
        "using the retrieved exercise context. Use only retrieved exercise context unless "
        "it is explicitly impossible. Do not invent exercise IDs. Return one complete "
        "JSON object matching the TrainingPlanRAGResponse schema. Do not return a single "
        "exercise object or an array of exercises. Return JSON only, with no markdown and "
        "no explanatory text outside the JSON object. The top-level object must include "
        "status, plan_type, user_id, summary, days, injury_warnings, and sources. Include "
        "reasons for chosen exercises and injury warnings when relevant."
    )
    user_prompt = (
        "Create a structured training plan from this user payload and retrieved context.\n\n"
        "Training response rules:\n"
        "- Return exactly one complete top-level JSON object.\n"
        "- Do not return a single exercise object.\n"
        "- Do not return markdown, code fences, or text outside JSON.\n"
        "- Top-level object must include status, plan_type, user_id, summary, days, injury_warnings, sources.\n"
        "- Every exercise must include exercise_id and source_id from the retrieved context.\n\n"
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
        "constraints, dislikes, and preferences from the user payload. If target_macros "
        "or normalized_target_macros are provided, copy them into macro_targets exactly "
        "or nearly exactly. If they are missing, estimate reasonable macro targets from "
        "the user's goal and profile. Do not return an unrealistically low-calorie daily "
        "plan. If calories or macros are too low, increase quantity or add more retrieved "
        "foods. Return one complete JSON object matching the NutritionPlanRAGResponse "
        "schema. Do not return a single food object or an array of meals. Return JSON "
        "only, with no markdown and no explanatory text outside the JSON object. The "
        "top-level object must include status, plan_type, user_id, daily_calorie_target, "
        "macro_targets, meals, allergy_warnings, and sources. Every food item must "
        "include food_id and source_id. source_id must equal the same food ID value "
        "unless the retrieved context uses a string ID. Include reasons for chosen foods "
        "and allergy warnings when relevant."
    )
    user_prompt = (
        "Create a structured nutrition plan from this user payload and retrieved context.\n\n"
        "Nutrition quality rules:\n"
        "- Use only foods from the retrieved food context.\n"
        "- Do not invent food IDs.\n"
        "- Avoid allergens and disliked foods from the user payload.\n"
        "- If normalized_target_macros is present, macro_targets must reflect those requested values.\n"
        "- If estimated_target_calories is present, the returned meals should provide a realistic full daily plan close to that target.\n"
        "- total_daily, if included, must be close to the sum of all returned meal foods.\n"
        "- If the plan is too low in calories or macros, increase quantity or add more retrieved foods.\n\n"
        "Schema rule for foods:\n"
        "- Every food in meals[].foods[] must include source_id.\n"
        "- Missing source_id makes the response invalid.\n"
        "- source_id must equal food_id unless the retrieved context uses a string ID.\n"
        "- Return exactly one complete top-level NutritionPlanRAGResponse JSON object.\n"
        "- Do not include markdown, code fences, or text outside JSON.\n"
        "- Top-level object must include status, plan_type, user_id, daily_calorie_target, macro_targets, meals, allergy_warnings, sources.\n"
        "- Example food item:\n"
        '{ "food_id": 16, "name": "Grilled Chicken Breast", "serving_size": "100g", "quantity": 2, "calories": 330, "protein": 62, "carbs": 0, "fat": 7.2, "reason": "Lean high-protein source from retrieved context.", "source_id": 16 }\n\n'
        "Required JSON shape:\n"
        "{\n"
        '  "status": "success",\n'
        '  "plan_type": "nutrition",\n'
        '  "user_id": "from user payload",\n'
        '  "daily_calorie_target": 2200,\n'
        '  "macro_targets": {"protein_g": 150, "carbs_g": 220, "fat_g": 70},\n'
        '  "total_daily": {"calories": 2190, "protein": 150, "carbs": 220, "fat": 70},\n'
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
