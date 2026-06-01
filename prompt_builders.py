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


def build_modify_training_plan_prompt(user_payload: dict, exercise_context: str) -> List[Dict[str, str]]:
    system_prompt = (
        "You are FitMind's RAG training-plan modification planner. Modify an existing "
        "training plan safely instead of generating a brand-new plan from zero. Use the "
        "current plan, user modification request, injuries, level, goal, weak points, "
        "and retrieved exercise context. Preserve useful unaffected days and exercises. "
        "Replace risky or unsuitable exercises with safer retrieved alternatives. Do not "
        "invent exercise IDs. For shoulder pain or shoulder injury, avoid overhead "
        "pressing and avoid direct shoulder-isolation replacements such as lateral "
        "raises, front raises, upright rows, rear-delt raises, face pulls, or shoulder presses unless "
        "the item is explicitly a very light rehab-safe drill. If no safe retrieved "
        "replacement exists, remove or reduce the risky exercise and recommend coach or "
        "physio review instead of forcing a risky replacement. Return one complete JSON "
        "object matching the "
        "ModifiedTrainingPlanRAGResponse schema. Return JSON only, with no markdown and "
        "no explanatory text outside the JSON object."
    )
    user_prompt = (
        "Modify the current training plan from this user payload and retrieved context.\n\n"
        "Modification rules:\n"
        "- Return exactly one complete top-level JSON object.\n"
        "- Do not return a single exercise object or only a list of replacements.\n"
        "- Keep the existing plan structure where it is safe and useful.\n"
        "- If an exercise may aggravate an injury or pain note, replace it with a safer retrieved exercise.\n"
        "- Shoulder pain rule: do not replace Shoulder Press with Lateral Raise, Front Raise, Upright Row, Rear Delt Raise, Face Pull, Arnold Press, or another direct shoulder-isolation/overhead movement as the main solution.\n"
        "- Shoulder pain during pressing rule: avoid upper-body pressing replacements such as push-ups, bench press, incline press, chest press, dips, or dumbbell press variations unless the user explicitly says they are pain-free.\n"
        "- For shoulder pain, prefer retrieved alternatives that do not directly stress the injured shoulder. If the old exercise is a shoulder exercise and no safe retrieved alternative exists, remove it or reduce it and add a clear injury warning.\n"
        "- For shoulder pain, add injury_warnings that mention avoiding overhead pressing and stopping movements that provoke pain.\n"
        "- For replacement or added exercises, exercise_id and source_id must come from the retrieved context.\n"
        "- If an existing exercise has no reliable ID, replace it with or map it to a retrieved exercise rather than inventing an ID.\n"
        "- Include concrete changes_summary entries that explain what changed and why.\n"
        "- Include injury_warnings when the user has injuries or pain-related notes.\n"
        "- Top-level object must include status, plan_type, user_id, summary, modified_plan, changes_summary, injury_warnings, sources.\n\n"
        "Required JSON shape:\n"
        "{\n"
        '  "status": "success",\n'
        '  "plan_type": "training_modification",\n'
        '  "user_id": "from user payload",\n'
        '  "summary": "short explanation of the modification",\n'
        '  "modified_plan": {\n'
        '    "schedule": [\n'
        '      {"day": "Day 1", "focus": "Push", "exercises": [\n'
        '        {"exercise_id": 1, "name": "Exercise name", "sets": 3, "reps": "8-12", "rest_seconds": 90, "intensity": "moderate", "reason": "why kept or replaced", "source_id": 1}\n'
        "      ]}\n"
        "    ]\n"
        "  },\n"
        '  "changes_summary": ["Replaced risky pressing movement with a safer retrieved alternative."],\n'
        '  "recommendations": ["Stop any movement that causes sharp pain."],\n'
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


def build_modify_nutrition_plan_prompt(user_payload: dict, food_context: str) -> List[Dict[str, str]]:
    system_prompt = (
        "You are FitMind's RAG nutrition-plan modification planner. Modify the existing "
        "nutrition plan safely and practically instead of generating a plan from zero. "
        "Use the current plan, modification request, target macros, user goal, level, "
        "meal count, allergies, liked foods, disliked foods, and retrieved food context. "
        "Use only retrieved food IDs for foods you add or replace. Do not invent food "
        "IDs. Remove or replace allergy-conflicting foods. Preserve safe useful meals "
        "where possible, but make the resulting daily meals realistic and close to the "
        "requested macro targets. Return one complete JSON object matching the "
        "ModifiedNutritionPlanRAGResponse schema. Return JSON only, with no markdown "
        "and no explanatory text outside the JSON object."
    )
    user_prompt = (
        "Modify the current nutrition plan from this user payload and retrieved food context.\n\n"
        "Modification rules:\n"
        "- Return exactly one complete top-level JSON object.\n"
        "- Do not return only a meal array or a single food object.\n"
        "- Use only foods from the retrieved food context for replacements or additions.\n"
        "- Every food in modified_plan.daily_meals[].foods[] must include food_id and source_id from the retrieved context.\n"
        "- source_id must equal food_id unless the retrieved context uses a string ID.\n"
        "- Do not include foods that conflict with allergies. If allergies include peanuts, do not include peanut, peanut butter, or peanut-containing foods.\n"
        "- Avoid disliked foods such as fish unless no safe macro-appropriate option exists; explain any unavoidable compromise in changes_summary.\n"
        "- Prefer liked foods such as chicken or yogurt when they fit macros and allergies.\n"
        "- Preserve target_macros as much as possible. If exact macro matching is impossible, explain that in changes_summary.\n"
        "- Do not make total_daily extremely below the requested calorie or macro target. Add retrieved foods or increase quantities if needed.\n"
        "- Include allergy_warnings when allergies or unsafe removed foods are relevant.\n"
        "- Include sources only for retrieved foods actually used in the modified plan.\n\n"
        "Required JSON shape:\n"
        "{\n"
        '  "status": "success",\n'
        '  "plan_type": "nutrition_modification",\n'
        '  "user_id": "from user payload",\n'
        '  "summary": "short explanation of the nutrition changes",\n'
        '  "modified_plan": {\n'
        '    "daily_meals": [\n'
        '      {"meal": "Breakfast", "foods": [\n'
        '        {"food_id": 1, "name": "Food name", "serving_size": "100g", "quantity": 1, "calories": 120, "protein": 20, "carbs": 4, "fat": 2, "reason": "why kept or selected", "source_id": 1}\n'
        "      ]}\n"
        "    ],\n"
        '    "total_daily": {"calories": 1800, "protein": 140, "carbs": 180, "fat": 55},\n'
        '    "daily_calorie_target": 1775,\n'
        '    "macro_targets": {"protein_g": 140, "carbs_g": 180, "fat_g": 55}\n'
        "  },\n"
        '  "changes_summary": ["Removed peanut-containing food and replaced it with a retrieved high-protein option."],\n'
        '  "recommendations": ["Continue avoiding peanut-containing foods."],\n'
        '  "allergy_warnings": ["Peanut allergy noted; peanut-containing foods were avoided."],\n'
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
