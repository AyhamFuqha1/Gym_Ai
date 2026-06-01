# main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import os
import hashlib
import pymysql
import chromadb
import time
import json
import traceback
import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import settings
from llm_client import call_llm_chat
from prompt_builders import (
    build_modify_nutrition_plan_prompt,
    build_modify_training_plan_prompt,
    build_nutrition_plan_prompt,
    build_training_plan_prompt,
)
from rag_schemas import (
    modified_nutrition_plan_response_schema,
    modified_training_plan_response_schema,
    nutrition_plan_response_schema,
    training_plan_response_schema,
)
from response_parsers import (
    parse_modified_nutrition_plan_response,
    parse_modified_training_plan_response,
    parse_nutrition_plan_response,
    parse_training_plan_response,
)
import re
import copy

logging.basicConfig(level=logging.DEBUG if settings.DEBUG else logging.INFO)
rag_logger = logging.getLogger("fitmind_ai.rag")

# =====================
# 🧠 EMBEDDINGS
# =====================
embeddings = None

def get_embeddings():
    global embeddings
    if embeddings is None:
        if not settings.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is required for embedding operations")
        embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.GEMINI_EMBEDDING_MODEL,
            google_api_key=settings.GEMINI_API_KEY
        )
    return embeddings

# =====================
# 🗄️ MYSQL CONFIG
# =====================
DB_CONFIG = settings.DB_CONFIG

# =====================
# 🧠 CHROMA DB - Collections
# =====================
client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
exercises_collection = client.get_or_create_collection(name=settings.EXERCISES_COLLECTION)
nutrition_collection = client.get_or_create_collection(name=settings.NUTRITION_COLLECTION)

# =====================
# 📌 MANIFEST & CHECKPOINT
# =====================
EXERCISES_MANIFEST = settings.EXERCISES_MANIFEST
NUTRITION_MANIFEST = settings.NUTRITION_MANIFEST
CHECKPOINT_FILE = settings.CHECKPOINT_FILE

# =====================
# 🔌 DB CONNECTION
# =====================
def get_db():
    return pymysql.connect(
        **DB_CONFIG,
        cursorclass=pymysql.cursors.DictCursor
    )

# =====================
# 📥 FETCH EXERCISES DATA
# =====================
def fetch_exercises_by_ids(ids=None):
    conn = get_db()
    cursor = conn.cursor()
    
    base_query = """
        SELECT 
            e.id,
            e.name,
            e.difficulty_level,
            e.instructions,
            e.common_mistakes,
            e.video_url,
            g.name AS exercise_category,
            g.muscle_group,
            g.description AS exercise_description,
            CONCAT(
                'Exercise: ', e.name, '\\n',
                'Muscle Group: ', g.muscle_group, '\\n',
                'Difficulty: ', e.difficulty_level, '\\n',
                'How to do it: ', IFNULL(e.instructions, 'Not specified'), '\\n',
                'Avoid: ', IFNULL(e.common_mistakes, 'Not specified'), '\\n',
                'About: ', IFNULL(g.description, 'No description')
            ) AS embedding_text
        FROM exercises e
        JOIN general_exercises g ON e.general_exercise_id = g.id
    """
    
    if ids:
        placeholders = ','.join(['%s'] * len(ids))
        cursor.execute(f"{base_query} WHERE e.id IN ({placeholders})", ids)
    else:
        cursor.execute(base_query)
    
    rows = cursor.fetchall()
    conn.close()
    return rows

def fetch_all_exercise_ids():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM exercises")
    rows = cursor.fetchall()
    conn.close()
    return [str(row['id']) for row in rows]

def table_has_column(cursor, table_name, column_name):
    cursor.execute("""
        SELECT COUNT(*) AS column_count
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = %s
          AND TABLE_NAME = %s
          AND COLUMN_NAME = %s
    """, (DB_CONFIG["database"], table_name, column_name))
    row = cursor.fetchone()
    return bool(row and int(row.get("column_count", 0)) > 0)

def fetch_changed_exercises(since_time):
    conn = get_db()
    cursor = conn.cursor()

    base_query = """
        SELECT 
            e.id,
            e.name,
            e.difficulty_level,
            e.instructions,
            e.common_mistakes,
            e.video_url,
            g.name AS exercise_category,
            g.muscle_group,
            g.description AS exercise_description,
            CONCAT(
                'Exercise: ', e.name, '\\n',
                'Muscle Group: ', g.muscle_group, '\\n',
                'Difficulty: ', e.difficulty_level, '\\n',
                'How to do it: ', IFNULL(e.instructions, 'Not specified'), '\\n',
                'Avoid: ', IFNULL(e.common_mistakes, 'Not specified'), '\\n',
                'About: ', IFNULL(g.description, 'No description')
            ) AS embedding_text
        FROM exercises e
        JOIN general_exercises g ON e.general_exercise_id = g.id
    """

    where_clauses = []
    params = []
    try:
        timestamp_checks = [
            ("e", "exercises", "updated_at"),
            ("e", "exercises", "created_at"),
            ("g", "general_exercises", "updated_at"),
            ("g", "general_exercises", "created_at"),
        ]
        for alias, table_name, column_name in timestamp_checks:
            if table_has_column(cursor, table_name, column_name):
                where_clauses.append(f"{alias}.{column_name} > %s")
                params.append(since_time)
    except Exception:
        where_clauses = []
        params = []

    # Some older FitMind databases do not expose updated_at/created_at on
    # exercise tables. In that case, fall back to scanning exercises so sync
    # remains correct instead of silently missing changed records.
    if where_clauses:
        base_query += " WHERE (" + " OR ".join(where_clauses) + ")"

    cursor.execute(base_query, params)
    rows = cursor.fetchall()
    conn.close()
    return rows

# =====================
# 📥 FETCH NUTRITION DATA
# =====================
def fetch_nutrition_by_ids(ids=None):
    conn = get_db()
    cursor = conn.cursor()
    
    base_query = """
        SELECT 
            f.id,
            f.name,
            f.calories,
            f.protein,
            f.carbs,
            f.fat,
            f.serving_size,
            gn.category_name,
            gn.description AS nutrition_description,
            CONCAT(
                'Food: ', f.name, '\\n',
                'Category: ', gn.category_name, '\\n',
                'Serving Size: ', f.serving_size, '\\n',
                'Calories: ', f.calories, ' cal\\n',
                'Protein: ', f.protein, 'g\\n',
                'Carbs: ', f.carbs, 'g\\n',
                'Fat: ', f.fat, 'g\\n',
                'Description: ', IFNULL(gn.description, 'No description')
            ) AS embedding_text
        FROM foods f
        JOIN general_nutrition gn ON f.general_nutrition_id = gn.id
    """
    
    if ids:
        placeholders = ','.join(['%s'] * len(ids))
        cursor.execute(f"{base_query} WHERE f.id IN ({placeholders})", ids)
    else:
        cursor.execute(base_query)
    
    rows = cursor.fetchall()
    conn.close()
    return rows

def fetch_all_nutrition_ids():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM foods")
    rows = cursor.fetchall()
    conn.close()
    return [str(row['id']) for row in rows]

def fetch_changed_nutrition(since_time):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            f.id,
            f.name,
            f.calories,
            f.protein,
            f.carbs,
            f.fat,
            f.serving_size,
            gn.category_name,
            gn.description AS nutrition_description,
            CONCAT(
                'Food: ', f.name, '\\n',
                'Category: ', gn.category_name, '\\n',
                'Serving Size: ', f.serving_size, '\\n',
                'Calories: ', f.calories, ' cal\\n',
                'Protein: ', f.protein, 'g\\n',
                'Carbs: ', f.carbs, 'g\\n',
                'Fat: ', f.fat, 'g\\n',
                'Description: ', IFNULL(gn.description, 'No description')
            ) AS embedding_text
        FROM foods f
        JOIN general_nutrition gn ON f.general_nutrition_id = gn.id
        WHERE (f.updated_at > %s OR f.created_at > %s)
    """, (since_time, since_time))
    rows = cursor.fetchall()
    conn.close()
    return rows

# =====================
# ITEM-LEVEL DOCUMENT STRATEGY
# =====================
# This service indexes structured database records, not long PDFs/articles.
# One exercise row becomes one Chroma document and one food row becomes one
# Chroma document. We intentionally do not use arbitrary character chunking.
# Retrieval quality comes from deterministic, rich, structured item text.

def safe_str(value, default=""):
    if value is None:
        return default
    return str(value).strip()

def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default

def csv_tags(tags):
    cleaned = []
    seen = set()
    for tag in tags or []:
        tag = normalize_label(tag)
        if tag and tag not in seen:
            seen.add(tag)
            cleaned.append(tag)
    return ",".join(cleaned)

def normalize_label(value):
    value = safe_str(value).lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[_\-/]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    aliases = {
        "protien": "protein",
        "protein": "protein",
        "healty fat": "healthy_fat",
        "healthy fat": "healthy_fat",
        "healthy fats": "healthy_fat",
        "vegetables": "vegetable",
        "vegetable": "vegetable",
        "fruits": "fruit",
        "fruit": "fruit",
        "carbohydrates": "carbohydrate",
        "carbs": "carbohydrate",
        "upper body": "upper_body",
        "lower body": "lower_body",
        "full body": "full_body",
    }
    if value in aliases:
        return aliases[value]

    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value

def normalize_food_category(category):
    category = normalize_label(category)
    aliases = {
        "seafoods": "seafood",
        "proteins": "protein",
        "healthy_fats": "healthy_fat",
        "healty_fat": "healthy_fat",
    }
    return aliases.get(category, category or "unknown")

def infer_normalized_muscle_group(raw_group="", exercise_name="", category="", description=""):
    checks = [
        ("chest", ["chest", "bench press", "push up", "push-up", "fly", "pec", "crossover"]),
        ("back", ["back", "row", "pulldown", "pull up", "pull-up", "lat"]),
        ("shoulders", ["shoulder", "lateral raise", "front raise", "rear delt", "face pull", "arnold press"]),
        ("biceps", ["bicep", "curl", "hammer curl", "concentration curl"]),
        ("triceps", ["tricep", "pushdown", "kickback", "extension", "close grip", "dip"]),
        ("legs", ["leg", "squat", "lunge", "deadlift", "hamstring", "quad", "calf", "step up"]),
        ("core", ["core", "plank", "dead bug", "crunch", "knee raise", "mountain climber", "russian twist"]),
        ("cardio", ["cardio", "jump rope", "burpee", "high knees", "jumping jack", "rowing machine"]),
        ("arms", ["arms", "arm"]),
    ]

    specific_labels = {label for label, _ in checks}
    generic_labels = {"upper_body", "lower_body", "full_body", "general"}

    def direct_label(value):
        normalized = normalize_label(value)
        if normalized in specific_labels or normalized in generic_labels:
            return normalized
        return None

    def keyword_match(text):
        text = safe_str(text).lower()
        if ("chest supported" in text or "chest-supported" in text) and "row" in text:
            return "back"
        for label, keywords in checks:
            if any(keyword in text for keyword in keywords):
                return label
        return None

    # Trust explicit DB/general-exercise labels before parsing the exercise
    # name. This prevents names like "Chest Supported Row" from overriding a
    # Back category or muscle group.
    for value in [raw_group, category]:
        label = direct_label(value)
        if label in specific_labels:
            return label

    for value in [raw_group, category, description]:
        db_match = keyword_match(value)
        if db_match:
            return db_match

    name_match = keyword_match(exercise_name)
    if name_match:
        return name_match

    for value in [raw_group, category]:
        label = direct_label(value)
        if label:
            return label

    normalized_raw = normalize_label(raw_group)
    return normalized_raw or normalize_label(category) or "general"

def infer_body_area(normalized_muscle_group, raw_group=""):
    raw = normalize_label(raw_group)
    if normalized_muscle_group in {"chest", "back", "shoulders", "biceps", "triceps", "arms"}:
        return "upper_body"
    if normalized_muscle_group == "legs":
        return "lower_body"
    if normalized_muscle_group in {"core", "cardio"}:
        return normalized_muscle_group
    if normalized_muscle_group in {"upper_body", "lower_body", "full_body"}:
        return normalized_muscle_group
    if raw in {"upper_body", "lower_body", "full_body"}:
        return raw
    return "general"

def infer_exercise_goal_tags(name, normalized_muscle_group, body_area, difficulty):
    tags = ["strength", "muscle_gain"]
    difficulty = normalize_label(difficulty)
    if difficulty:
        tags.append(difficulty)
    if body_area == "upper_body":
        tags.append("upper_body_training")
    if body_area == "lower_body":
        tags.append("lower_body_training")
    if normalized_muscle_group in {"chest", "shoulders", "triceps"}:
        tags.append("push_day")
    if normalized_muscle_group in {"back", "biceps"}:
        tags.append("pull_day")
    if normalized_muscle_group == "legs":
        tags.extend(["leg_day", "lower_body"])
    if normalized_muscle_group == "core":
        tags.append("core_training")
    if normalized_muscle_group == "cardio":
        tags.extend(["fat_loss", "conditioning"])

    name_l = safe_str(name).lower()
    if any(word in name_l for word in ["push up", "plank", "burpee", "mountain climber", "jumping jack"]):
        tags.append("bodyweight")
    if any(word in name_l for word in ["machine", "cable", "smith"]):
        tags.append("machine_or_cable")
    return tags

def infer_exercise_search_phrases(name, normalized_muscle_group, body_area, difficulty, goal_tags):
    phrases = []
    difficulty = normalize_label(difficulty)
    if normalized_muscle_group and normalized_muscle_group != "general":
        phrases.append(f"{difficulty or 'beginner'} {normalized_muscle_group} workout")
        phrases.append(f"{normalized_muscle_group} exercise")
    if "push_day" in goal_tags:
        phrases.extend(["push day exercise", "upper body training"])
    if "pull_day" in goal_tags:
        phrases.extend(["pull day exercise", "upper body training"])
    if "leg_day" in goal_tags:
        phrases.extend(["leg day exercise", "lower body training"])
    if body_area == "core":
        phrases.append("core stability training")
    if "bodyweight" in goal_tags:
        phrases.append("bodyweight workout")
    return phrases

def infer_exercise_caution_hints(name, normalized_muscle_group):
    text = safe_str(name).lower()
    hints = []
    if normalized_muscle_group in {"chest", "shoulders"} or any(word in text for word in ["press", "fly", "push up"]):
        hints.append("use caution with shoulder discomfort")
    if normalized_muscle_group in {"biceps", "triceps"} or any(word in text for word in ["curl", "pushdown", "kickback"]):
        hints.append("use caution with elbow or wrist discomfort")
    if normalized_muscle_group == "legs" or any(word in text for word in ["squat", "lunge", "leg press", "step up"]):
        hints.append("use caution with knee discomfort")
    if any(word in text for word in ["deadlift", "bent over", "row"]):
        hints.append("use caution with lower-back discomfort")
    if hints:
        hints.append("not medical advice")
    return hints

def infer_food_goal_tags(calories, protein, carbs, fat, category):
    tags = []
    calories = safe_float(calories)
    protein = safe_float(protein)
    carbs = safe_float(carbs)
    fat = safe_float(fat)
    category = normalize_food_category(category)

    if calories > 0:
        tags.append("balanced")
        protein_calorie_ratio = (protein * 4) / calories if calories else 0
        if calories <= 120:
            tags.append("low_calorie")
        if calories <= 220 and fat <= 10:
            tags.append("fat_loss")
        if protein >= 12 or protein_calorie_ratio >= 0.25:
            tags.append("high_protein")
        if protein >= 8 and calories >= 100:
            tags.append("muscle_gain")
        if fat <= 3:
            tags.append("low_fat")
    if category == "healthy_fat" or fat >= 12:
        tags.append("healthy_fat")
    if category == "vegetable":
        tags.extend(["low_calorie", "fiber"])
    if category == "fruit":
        tags.extend(["snack", "carbohydrate"])
    if category in {"protein", "seafood"}:
        tags.append("lean_protein" if fat <= 10 else "protein")
    if carbs >= 20:
        tags.append("carbohydrate")
    return tags

def infer_meal_role_tags(food_name, category, calories, protein, carbs, fat):
    name = safe_str(food_name).lower()
    category = normalize_food_category(category)
    calories = safe_float(calories)
    protein = safe_float(protein)
    fat = safe_float(fat)
    tags = []

    if any(word in name for word in ["oat", "egg", "yogurt", "banana", "bread", "toast"]):
        tags.append("breakfast")
    if category in {"protein", "seafood", "vegetable", "carbohydrate"} or any(
        word in name for word in ["chicken", "beef", "turkey", "rice", "potato", "fish", "salmon", "tuna"]
    ):
        tags.extend(["lunch", "dinner"])
    if category in {"fruit", "healthy_fat"} or calories <= 120 or any(
        word in name for word in ["almond", "walnut", "peanut", "yogurt", "banana"]
    ):
        tags.append("snack")
    if protein >= 20:
        tags.extend(["lunch", "dinner"])
    if fat >= 20:
        tags.append("snack")
    return tags or ["meal_item"]

def mostly_repeated_or_random_text(value):
    text = safe_str(value).lower()
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return True
    if re.search(r"(.)\1{4,}", compact):
        return True
    if any(token in compact for token in ["asdf", "qwer", "testtest", "aaaaaaaa", "111111"]):
        return True

    letters = re.sub(r"[^a-z\u0600-\u06ff]", "", compact)
    if len(letters) >= 8:
        unique_ratio = len(set(letters)) / len(letters)
        vowels = sum(ch in "aeiou" for ch in letters)
        if unique_ratio < 0.25:
            return True
        if re.fullmatch(r"[a-z]+", letters) and vowels == 0 and len(set(letters)) <= 5:
            return True
    return False

def is_junk_item(name, *details):
    name_clean = safe_str(name)
    normalized_name = normalize_label(name_clean)
    if normalized_name in {"", "test", "testing", "dummy", "sample", "none", "null"}:
        return True
    if len(name_clean) < 2:
        return True
    if mostly_repeated_or_random_text(name_clean) and len(name_clean) >= 5:
        return True

    useful_details = [safe_str(detail) for detail in details if safe_str(detail)]
    if useful_details:
        noisy_details = [
            detail for detail in useful_details
            if len(detail) >= 8 and mostly_repeated_or_random_text(detail)
        ]
        if noisy_details and len(noisy_details) == len(useful_details):
            return True
    return False

def exercise_search_quality(row):
    if is_junk_item(row.get("name"), row.get("instructions"), row.get("common_mistakes")):
        return "junk"
    instruction = safe_str(row.get("instructions"))
    if len(instruction) < 12:
        return "partial"
    return "good"

def food_search_quality(row):
    if is_junk_item(row.get("name"), row.get("nutrition_description")):
        return "junk"
    calories = safe_float(row.get("calories"))
    if calories <= 0:
        return "partial"
    return "good"

def build_exercise_item_document(row):
    raw_name = safe_str(row.get("name"), "Unnamed exercise")
    raw_group = safe_str(row.get("muscle_group"), "General")
    category = safe_str(row.get("exercise_category"))
    description = safe_str(row.get("exercise_description"), "No description")
    difficulty = safe_str(row.get("difficulty_level"), "beginner")
    normalized_muscle = infer_normalized_muscle_group(raw_group, raw_name, category, description)
    body_area = infer_body_area(normalized_muscle, raw_group)
    goal_tags = infer_exercise_goal_tags(raw_name, normalized_muscle, body_area, difficulty)
    search_phrases = infer_exercise_search_phrases(raw_name, normalized_muscle, body_area, difficulty, goal_tags)
    caution_hints = infer_exercise_caution_hints(raw_name, normalized_muscle)

    return "\n".join([
        "Item strategy: one exercise database record is indexed as one Chroma document.",
        "Chunking: item-level structured record; no arbitrary character chunking.",
        "Type: strength exercise",
        f"Exercise name: {raw_name}",
        f"Primary muscle group: {normalized_muscle}",
        f"Raw muscle group: {raw_group}",
        f"Body area: {body_area}",
        f"Difficulty: {difficulty}",
        f"Instructions: {safe_str(row.get('instructions'), 'Not specified')}",
        f"Common mistakes: {safe_str(row.get('common_mistakes'), 'Not specified')}",
        f"Category: {category or raw_group}",
        f"Category description: {description}",
        f"Goal tags: {csv_tags(goal_tags)}",
        f"Search phrases: {', '.join(search_phrases) if search_phrases else raw_name}",
        f"Injury caution hints: {', '.join(caution_hints) if caution_hints else 'none inferred'}",
    ])

def build_food_item_document(row):
    raw_name = safe_str(row.get("name"), "Unnamed food")
    category = normalize_food_category(row.get("category_name"))
    calories = safe_float(row.get("calories"))
    protein = safe_float(row.get("protein"))
    carbs = safe_float(row.get("carbs"))
    fat = safe_float(row.get("fat"))
    goal_tags = infer_food_goal_tags(calories, protein, carbs, fat, category)
    meal_roles = infer_meal_role_tags(raw_name, category, calories, protein, carbs, fat)

    return "\n".join([
        "Item strategy: one food database record is indexed as one Chroma document.",
        "Chunking: item-level structured record; no arbitrary character chunking.",
        "Type: food",
        f"Food name: {raw_name}",
        f"Normalized category: {category}",
        f"Raw category: {safe_str(row.get('category_name'), 'unknown')}",
        f"Serving size: {safe_str(row.get('serving_size'), 'Not specified')}",
        f"Calories: {calories:g}",
        f"Protein: {protein:g}g",
        f"Carbs: {carbs:g}g",
        f"Fat: {fat:g}g",
        f"Description: {safe_str(row.get('nutrition_description'), 'No description')}",
        f"Goal tags: {csv_tags(goal_tags)}",
        f"Meal role tags: {csv_tags(meal_roles)}",
    ])

def exercise_row_to_text(row):
    return build_exercise_item_document(row)

def nutrition_row_to_text(row):
    return build_food_item_document(row)

def vector_id(data_type: str, row_id):
    if data_type in {"exercise", "exercises"}:
        return f"exercise:{row_id}"
    if data_type in {"food", "nutrition"}:
        return f"food:{row_id}"
    return str(row_id)

# =====================
# 🔐 HASH FUNCTION
# =====================
def hash_row(payload):
    if not isinstance(payload, str):
        payload = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def row_fingerprint(embedding_text: str, metadata: Dict[str, Any]):
    return hash_row({
        "document": embedding_text,
        "metadata": metadata,
    })

def manifest_has_legacy_ids(manifest_file: str, prefix: str):
    manifest = load_manifest(manifest_file)
    if not manifest:
        return False
    return any(not str(row_id).startswith(f"{prefix}:") for row_id in manifest.keys())

# =====================
# 📂 LOAD/SAVE MANIFEST
# =====================
def load_manifest(file_path):
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_manifest(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# =====================
# 📂 LOAD/SAVE CHECKPOINT
# =====================
def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return {"exercises_batch": 0, "nutrition_batch": 0, "last_sync_time": None}
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_checkpoint(data):
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# =====================
# 🧠 EMBED
# =====================
def embed(text):
    return get_embeddings().embed_query(text)

# =====================
# 📦 PROCESS BATCH (عام)
# =====================
def process_batch(rows, collection, manifest_file, row_to_text_func, get_metadata_func, batch_num, total_batches, data_type, force_upsert=False):
    print(f"  📦 Processing {data_type} batch {batch_num}/{total_batches} ({len(rows)} items)...")
    
    old_manifest = load_manifest(manifest_file)
    new_manifest = dict(old_manifest)
    added, updated = 0, 0
    
    for row in rows:
        embedding_text = row_to_text_func(row)
        metadata = get_metadata_func(row)
        row_hash = row_fingerprint(embedding_text, metadata)
        row_id = vector_id(data_type, row["id"])

        old_hash = old_manifest.get(row_id)
        new_manifest[row_id] = row_hash
        
        if old_hash is None:
            vector = embed(embedding_text)
            collection.upsert(
                ids=[row_id],
                documents=[embedding_text],
                embeddings=[vector],
                metadatas=[metadata]
            )
            added += 1
        elif force_upsert or old_hash != row_hash:
            vector = embed(embedding_text)
            collection.upsert(
                ids=[row_id],
                documents=[embedding_text],
                embeddings=[vector],
                metadatas=[metadata]
            )
            updated += 1
    
    save_manifest(manifest_file, new_manifest)
    print(f"     ✅ {data_type} batch done: +{added} added, 🔄{updated} updated")
    return added, updated

# =====================
# 📦 METADATA FUNCTIONS
# =====================
def get_exercise_metadata(row):
    raw_group = safe_str(row.get("muscle_group"), "General")
    category = safe_str(row.get("exercise_category"))
    description = safe_str(row.get("exercise_description"))
    normalized_muscle = infer_normalized_muscle_group(raw_group, row.get("name"), category, description)
    body_area = infer_body_area(normalized_muscle, raw_group)
    difficulty = safe_str(row.get("difficulty_level"), "beginner")
    goal_tags = infer_exercise_goal_tags(row.get("name"), normalized_muscle, body_area, difficulty)
    search_quality = exercise_search_quality(row)

    return {
        "id": row["id"],
        "type": "exercise",
        "name": safe_str(row.get("name"), "Unnamed exercise"),
        "muscle_group": raw_group,
        "normalized_muscle_group": normalized_muscle,
        "body_area": body_area,
        "difficulty": difficulty,
        "goal_tags": csv_tags(goal_tags),
        "search_quality": search_quality,
    }

def get_nutrition_metadata(row):
    category = safe_str(row.get("category_name"), "unknown")
    normalized_category = normalize_food_category(category)
    calories = safe_float(row.get("calories"))
    protein = safe_float(row.get("protein"))
    carbs = safe_float(row.get("carbs"))
    fat = safe_float(row.get("fat"))
    goal_tags = infer_food_goal_tags(calories, protein, carbs, fat, normalized_category)
    meal_role_tags = infer_meal_role_tags(row.get("name"), normalized_category, calories, protein, carbs, fat)
    search_quality = food_search_quality(row)

    return {
        "id": row["id"],
        "type": "food",
        "name": safe_str(row.get("name"), "Unnamed food"),
        "category": category,
        "normalized_category": normalized_category,
        "calories": calories,
        "protein": protein,
        "carbs": carbs,
        "fat": fat,
        "goal_tags": csv_tags(goal_tags),
        "meal_role_tags": csv_tags(meal_role_tags),
        "search_quality": search_quality,
    }

# =====================
# 🔄 SYNC FUNCTIONS
# =====================
def sync_exercises_to_vector(full_sync=False):
    print("🔄 Syncing Exercises...")
    total_added, total_updated = 0, 0

    if not full_sync and manifest_has_legacy_ids(EXERCISES_MANIFEST, "exercise"):
        print("   Legacy exercise vector ids detected; running one-time full sync migration")
        return sync_exercises_to_vector(full_sync=True)
    
    if full_sync:
        all_ids = fetch_all_exercise_ids()
        if not all_ids:
            print("   No exercises found")
            return 0, 0
        
        batch_size = getattr(settings, 'BATCH_SIZE', 50)
        batches = [all_ids[i:i+batch_size] for i in range(0, len(all_ids), batch_size)]
        total_batches = len(batches)
        
        for batch_num, batch_ids in enumerate(batches):
            rows = fetch_exercises_by_ids(batch_ids)
            if rows:
                added, updated = process_batch(
                    rows, exercises_collection, EXERCISES_MANIFEST,
                    exercise_row_to_text, get_exercise_metadata,
                    batch_num + 1, total_batches, "exercises",
                    force_upsert=full_sync
                )
                total_added += added
                total_updated += updated
            
            if batch_num < total_batches - 1:
                time.sleep(getattr(settings, 'SYNC_DELAY_SECONDS', 1))
    else:
        checkpoint = load_checkpoint()
        last_sync = checkpoint.get("last_sync_time")
        if last_sync:
            rows = fetch_changed_exercises(last_sync)
        else:
            return sync_exercises_to_vector(full_sync=True)
        
        if rows:
            added, updated = process_batch(
                rows, exercises_collection, EXERCISES_MANIFEST,
                exercise_row_to_text, get_exercise_metadata,
                1, 1, "exercises"
            )
            total_added, total_updated = added, updated
    
    print(f"   ✅ Exercises sync: +{total_added} added, 🔄{total_updated} updated")
    return total_added, total_updated

def sync_nutrition_to_vector(full_sync=False):
    print("🔄 Syncing Nutrition...")
    total_added, total_updated = 0, 0

    if not full_sync and manifest_has_legacy_ids(NUTRITION_MANIFEST, "food"):
        print("   Legacy nutrition vector ids detected; running one-time full sync migration")
        return sync_nutrition_to_vector(full_sync=True)
    
    if full_sync:
        all_ids = fetch_all_nutrition_ids()
        if not all_ids:
            print("   No nutrition items found")
            return 0, 0
        
        batch_size = getattr(settings, 'BATCH_SIZE', 50)
        batches = [all_ids[i:i+batch_size] for i in range(0, len(all_ids), batch_size)]
        total_batches = len(batches)
        
        for batch_num, batch_ids in enumerate(batches):
            rows = fetch_nutrition_by_ids(batch_ids)
            if rows:
                added, updated = process_batch(
                    rows, nutrition_collection, NUTRITION_MANIFEST,
                    nutrition_row_to_text, get_nutrition_metadata,
                    batch_num + 1, total_batches, "nutrition",
                    force_upsert=full_sync
                )
                total_added += added
                total_updated += updated
            
            if batch_num < total_batches - 1:
                time.sleep(getattr(settings, 'SYNC_DELAY_SECONDS', 1))
    else:
        checkpoint = load_checkpoint()
        last_sync = checkpoint.get("last_sync_time")
        if last_sync:
            rows = fetch_changed_nutrition(last_sync)
        else:
            return sync_nutrition_to_vector(full_sync=True)
        
        if rows:
            added, updated = process_batch(
                rows, nutrition_collection, NUTRITION_MANIFEST,
                nutrition_row_to_text, get_nutrition_metadata,
                1, 1, "nutrition"
            )
            total_added, total_updated = added, updated
    
    print(f"   ✅ Nutrition sync: +{total_added} added, 🔄{total_updated} updated")
    return total_added, total_updated

def prune_deleted_records(collection, manifest_file, current_ids):
    manifest = load_manifest(manifest_file)
    manifest_ids = set(manifest.keys())
    deleted_ids = manifest_ids - current_ids

    if deleted_ids:
        collection.delete(ids=sorted(deleted_ids))

    pruned_manifest = {
        row_id: manifest[row_id]
        for row_id in sorted(current_ids)
        if row_id in manifest
    }
    save_manifest(manifest_file, pruned_manifest)
    return len(deleted_ids)

def sync_all(full_sync=False):
    print("🔄 Starting Full Smart Sync...")
    start_time = time.time()
    
    exercises_added, exercises_updated = sync_exercises_to_vector(full_sync)
    nutrition_added, nutrition_updated = sync_nutrition_to_vector(full_sync)
    
    current_exercise_ids = {
        vector_id("exercise", row_id)
        for row_id in fetch_all_exercise_ids()
    }
    deleted_exercises = prune_deleted_records(
        exercises_collection,
        EXERCISES_MANIFEST,
        current_exercise_ids
    )
    
    current_nutrition_ids = {
        vector_id("food", row_id)
        for row_id in fetch_all_nutrition_ids()
    }
    deleted_nutrition = prune_deleted_records(
        nutrition_collection,
        NUTRITION_MANIFEST,
        current_nutrition_ids
    )
    
    save_checkpoint({
        "last_sync_time": datetime.now().isoformat(),
        "exercises_batch": 0,
        "nutrition_batch": 0
    })
    
    elapsed_time = time.time() - start_time
    print(f"\n✅ ALL SYNC DONE in {elapsed_time:.2f} seconds")
    print(f"   Exercises: +{exercises_added} added, 🔄{exercises_updated} updated, 🗑{deleted_exercises} deleted")
    print(f"   Nutrition: +{nutrition_added} added, 🔄{nutrition_updated} updated, 🗑{deleted_nutrition} deleted")
    
    return {
        "exercises": {"added": exercises_added, "updated": exercises_updated, "deleted": deleted_exercises},
        "nutrition": {"added": nutrition_added, "updated": nutrition_updated, "deleted": deleted_nutrition},
        "elapsed_seconds": elapsed_time
    }

# =====================
# 🔍 SEARCH FUNCTIONS
# =====================
def collection_query_count(collection, requested):
    try:
        count = collection.count()
        if count > 0:
            return min(requested, count)
    except Exception:
        pass
    return requested

def unpack_chroma_results(results):
    docs = results.get("documents", [[]])
    metas = results.get("metadatas", [[]])
    ids = results.get("ids", [[]])
    distances = results.get("distances", [[]])

    docs_row = docs[0] if docs else []
    metas_row = metas[0] if metas else []
    ids_row = ids[0] if ids else []
    distances_row = distances[0] if distances else []

    items = []
    for idx, (doc, meta) in enumerate(zip(docs_row, metas_row)):
        items.append({
            "document": doc,
            "metadata": meta or {},
            "id": ids_row[idx] if idx < len(ids_row) else None,
            "distance": distances_row[idx] if idx < len(distances_row) else None,
        })
    return items

def pack_chroma_results(items):
    return {
        "ids": [[item.get("id") for item in items]],
        "documents": [[item.get("document") for item in items]],
        "metadatas": [[item.get("metadata") or {} for item in items]],
        "distances": [[item.get("distance") for item in items]],
    }

def tag_string_contains(tags, wanted):
    wanted = normalize_label(wanted)
    if not wanted:
        return True
    tag_set = {
        normalize_label(tag)
        for tag in safe_str(tags).split(",")
        if normalize_label(tag)
    }
    return wanted in tag_set

def item_search_quality(meta):
    return safe_str(meta.get("search_quality"), "good").lower() or "good"

def exercise_item_matches_filters(item, filters):
    filters = filters or {}
    meta = item.get("metadata") or {}

    if filters.get("exclude_junk", True) and item_search_quality(meta) == "junk":
        return False

    difficulty = filters.get("difficulty")
    if difficulty and normalize_label(meta.get("difficulty")) != normalize_label(difficulty):
        return False

    muscle_group = filters.get("muscle_group")
    if muscle_group:
        wanted = normalize_label(muscle_group)
        candidates = {
            normalize_label(meta.get("muscle_group")),
            normalize_label(meta.get("normalized_muscle_group")),
            normalize_label(meta.get("body_area")),
        }
        if wanted not in candidates:
            return False

    goal = filters.get("goal")
    if goal and not tag_string_contains(meta.get("goal_tags"), goal):
        return False

    return True

def food_item_matches_filters(item, filters):
    filters = filters or {}
    meta = item.get("metadata") or {}

    if filters.get("exclude_junk", True) and item_search_quality(meta) == "junk":
        return False

    category = filters.get("category")
    if category:
        wanted = normalize_food_category(category)
        candidates = {
            normalize_food_category(meta.get("category")),
            normalize_food_category(meta.get("normalized_category")),
        }
        if wanted not in candidates:
            return False

    min_protein = filters.get("min_protein")
    if min_protein is not None and safe_float(meta.get("protein")) < safe_float(min_protein):
        return False

    max_calories = filters.get("max_calories")
    if max_calories is not None and safe_float(meta.get("calories")) > safe_float(max_calories):
        return False

    goal = filters.get("goal")
    if goal and not tag_string_contains(meta.get("goal_tags"), goal):
        return False

    return True

def filtered_collection_query(collection, query, n_results, filters, match_func):
    query_vector = embed(query)
    fetch_count = n_results
    if filters:
        fetch_count = max(n_results * 4, n_results + 10)
    fetch_count = collection_query_count(collection, fetch_count)

    if fetch_count <= 0:
        return pack_chroma_results([])

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=fetch_count
    )

    items = unpack_chroma_results(results)
    filtered_items = [
        item for item in items
        if match_func(item, filters)
    ][:n_results]
    return pack_chroma_results(filtered_items)

def search_exercises(query, n_results=10, filters=None):
    filters = dict(filters or {})
    filters.setdefault("exclude_junk", True)
    return filtered_collection_query(
        exercises_collection,
        query,
        n_results,
        filters,
        exercise_item_matches_filters,
    )

def search_nutrition(query, n_results=10, filters=None):
    filters = dict(filters or {})
    filters.setdefault("exclude_junk", True)
    return filtered_collection_query(
        nutrition_collection,
        query,
        n_results,
        filters,
        food_item_matches_filters,
    )

def legacy_search_exercises(query, n_results=10):
    query_vector = embed(query)
    results = exercises_collection.query(
        query_embeddings=[query_vector],
        n_results=n_results
    )
    return results

def legacy_search_nutrition(query, n_results=10):
    query_vector = embed(query)
    results = nutrition_collection.query(
        query_embeddings=[query_vector],
        n_results=n_results
    )
    return results

# =====================
# 📦 PYDANTIC MODELS
# =====================
class UserSummary(BaseModel):
    user_id: int
    level: str
    goal: str
    training_age_years: float
    injuries: List[str] = []
    weak_points: List[str] = []
    status: str = "progressing"
    progress_rate: float = 0.0
    consistency_score: float = 0.0
    weight: Optional[float] = None
    height: Optional[float] = None
    age: Optional[int] = None

class GenerateTrainingRequest(BaseModel):
    user_summary: UserSummary
    preferences: Dict[str, Any] = {}
    previous_plans: List[Dict[str, Any]] = []

class GenerateNutritionRequest(BaseModel):
    user_summary: UserSummary
    preferences: Dict[str, Any] = {}
    target_macros: Optional[Dict[str, float]] = None

class AnalyzeProgressRequest(BaseModel):
    user_summary: UserSummary
    progress_data: Dict[str, Any]
    current_plan_id: Optional[Union[int, str]] = None

class SearchRequest(BaseModel):
    query: Optional[str] = None
    n_results: Optional[int] = 10
    difficulty: Optional[str] = None
    muscle_group: Optional[str] = None
    category: Optional[str] = None
    min_protein: Optional[float] = None
    max_calories: Optional[float] = None
    goal: Optional[str] = None
    exclude_junk: Optional[bool] = True
    debug_context: Optional[bool] = False

class SmartModifyTrainingRequest(BaseModel):
    current_plan_id: Optional[Union[int, str]] = None
    current_plan: Dict[str, Any]
    user_summary: UserSummary
    user_feedback: Dict[str, Any] = {}
    modification_request: Optional[Union[Dict[str, Any], str]] = None

class SmartModifyNutritionRequest(BaseModel):
    current_plan_id: Optional[Union[int, str]] = None
    current_plan: Dict[str, Any]
    user_summary: UserSummary
    user_feedback: Dict[str, Any] = {}
    preferences: Dict[str, Any] = {}
    target_macros: Optional[Dict[str, Any]] = None
    modification_request: Optional[Union[Dict[str, Any], str]] = None

# =====================
# 🔧 SMART MODIFY FUNCTIONS
# =====================

def extract_requested_split(modification_text: str):
    if not modification_text:
        return []

    text = modification_text.strip()
    text = re.sub(r"\s+", " ", text)

    # خذ فقط جزء الـ Day split، بدون سحب الملاحظات اللاحقة داخل آخر يوم
    matches = re.findall(
        r"day\s*(\d+)\s*[:\-]?\s*(.*?)(?=\s*day\s*\d+\s*[:\-]?|$)",
        text,
        re.IGNORECASE,
    )

    parsed = []
    for day_num, raw_focus in matches:
        focus = raw_focus.strip(" .,-")
        # قص أي جملة ملاحظات بعد اسم العضلات
        focus = re.split(
            r"\.\s+|,\s*(?:avoid|keep|make|please|reduce|easier|harder)\b|\bavoid\b|\bkeep\b",
            focus,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" .,-")

        if focus:
            parsed.append({
                "day": int(day_num),
                "focus": focus
            })

    parsed.sort(key=lambda x: x["day"])
    return parsed


def normalize_focus_label(focus: str):
    f = str(focus or "").lower().strip()

    replacements = {
        "&": " and ",
        "/": " ",
        "-": " ",
        "_": " ",
    }
    for old, new in replacements.items():
        f = f.replace(old, new)

    f = re.sub(r"\s+", " ", f).strip()

    aliases = {
        "bi": "biceps",
        "tri": "triceps",
        "bi and tri": "biceps and triceps",
        "biceps triceps": "biceps and triceps",
        "arms": "biceps and triceps",
        "upper body": "upper",
        "lower body": "lower",
        "push day": "push",
        "pull day": "pull",
        "leg day": "legs",
        "shoulder day": "shoulders",
        "back day": "back",
        "chest day": "chest",
        "full body day": "full body",
    }

    return aliases.get(f, f)


def get_focus_keywords(focus: str):
    focus = normalize_focus_label(focus)

    keyword_map = {
        "chest": ["chest", "press", "fly", "pec"],
        "back": ["back", "row", "pulldown", "pull up", "lat"],
        "shoulders": ["shoulder", "lateral raise", "rear delt", "face pull"],
        "legs": ["leg", "squat", "lunge", "leg press", "leg curl", "hamstring", "quad", "calf"],
        "biceps": ["bicep", "curl"],
        "triceps": ["tricep", "pushdown", "extension", "kickback"],
        "biceps and triceps": ["bicep", "curl", "tricep", "pushdown", "extension", "kickback"],
        "push": ["chest", "press", "fly", "pec", "tricep", "pushdown", "extension"],
        "pull": ["back", "row", "pulldown", "lat", "bicep", "curl", "face pull"],
        "upper": ["chest", "press", "fly", "back", "row", "pulldown", "shoulder", "raise", "bicep", "curl", "tricep"],
        "lower": ["leg", "squat", "lunge", "leg press", "leg curl", "hamstring", "quad", "calf"],
        "full body": ["chest", "row", "leg", "core"],
        "core": ["plank", "dead bug", "crunch", "core", "ab", "mountain climber"],
        "chest and triceps": ["chest", "press", "fly", "pec", "tricep", "pushdown", "extension"],
        "back and biceps": ["back", "row", "pulldown", "lat", "bicep", "curl"],
    }

    if focus in keyword_map:
        return keyword_map[focus]

    parts = re.split(r"\band\b|,|\/", focus)
    parts = [normalize_focus_label(p.strip()) for p in parts if p.strip()]

    keywords = []
    for p in parts:
        if p in keyword_map:
            keywords.extend(keyword_map[p])
        else:
            keywords.append(p)

    deduped = []
    seen = set()
    for k in keywords:
        if k not in seen:
            seen.add(k)
            deduped.append(k)

    return deduped


def exercise_matches_focus(ex_name: str, focus: str):
    name = str(ex_name or "").lower().strip()
    keywords = get_focus_keywords(focus)
    return any(k in name for k in keywords)


def get_injury_blocked_keywords(pain_areas: list):
    injury_map = {
        "elbow": [
            "curl", "pushdown", "kickback", "skull crusher", "overhead extension",
            "close grip", "reverse grip", "arnold press", "shoulder press",
            "barbell curl", "ez bar curl", "concentration curl", "hammer curl",
            "cable bicep curl", "dumbbell bicep curl", "tricep", "dip", "push up"
        ],
        "shoulder": [
            "shoulder press", "arnold press", "upright row", "front raise",
            "lateral raise", "rear delt", "shoulder raise", "overhead press",
            "military press", "face pull", "bench press", "incline press", "pec deck", "fly", "push up"
        ],
        "knee": [
            "leg extension", "walking lunge", "jump squat", "bulgarian",
            "step up", "box step up", "deep squat"
        ],
        "lower back": [
            "romanian deadlift", "deadlift", "good morning", "bent over row"
        ],
        "back": [
            "romanian deadlift", "deadlift", "good morning", "bent over row"
        ],
        "wrist": [
            "barbell curl", "straight bar", "bench press", "push up"
        ],
    }

    blocked = set()
    for pain in pain_areas:
        pain_l = str(pain).strip().lower()
        for key, values in injury_map.items():
            if key in pain_l:
                blocked.update(v.lower() for v in values)

    return blocked


def build_queries_for_focus_dynamic(focus: str, pain_areas: list, difficulty: str, liked_exercises: list):
    normalized_focus = normalize_focus_label(focus)
    queries = []

    queries.append(f"{normalized_focus} exercise")

    if pain_areas:
        for pain in pain_areas:
            pain_l = str(pain).lower().strip()
            queries.append(f"safe {normalized_focus} exercise for {pain_l}")
            queries.append(f"{pain_l} friendly {normalized_focus} exercise")

    if difficulty == "too_hard":
        queries.append(f"beginner {normalized_focus} exercise")
        queries.append(f"easy {normalized_focus} exercise")
    elif difficulty == "too_easy":
        queries.append(f"advanced {normalized_focus} exercise")
        queries.append(f"challenging {normalized_focus} exercise")

    for liked in liked_exercises:
        liked_l = liked.lower().strip()
        if any(k in liked_l for k in get_focus_keywords(normalized_focus)):
            queries.append(liked)

    final_queries = []
    seen = set()
    for q in queries:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            final_queries.append(q)

    return final_queries[:10]


def collect_suggested_exercises(
    focus: str,
    search_queries,
    search_func,
    blocked_keywords=None,
    excluded_ids=None,
    excluded_names=None,
    target_count: int = 4,
):
    blocked_keywords = blocked_keywords or set()
    excluded_ids = excluded_ids or set()
    excluded_names = excluded_names or set()

    suggested = []
    seen_ids = set()
    seen_names = set()

    def matches_blocked(name: str) -> bool:
        name_l = str(name or "").strip().lower()
        return any(word in name_l for word in blocked_keywords)

    for query in search_queries:
        results = search_func(query, n_results=15)
        docs = results.get("documents", [[]])
        metas = results.get("metadatas", [[]])

        if not docs or not metas:
            continue

        for _, meta in zip(docs[0], metas[0]):
            ex_id = meta.get("id")
            ex_name = str(meta.get("name", "")).strip()
            ex_name_l = ex_name.lower()

            if not ex_id or not ex_name:
                continue
            if ex_id in seen_ids or ex_id in excluded_ids:
                continue
            if ex_name_l in seen_names or ex_name_l in excluded_names:
                continue
            if matches_blocked(ex_name):
                continue
            if not exercise_matches_focus(ex_name, focus):
                continue
            suggested.append({
                "id": ex_id,
                "name": ex_name,
                "muscle_group": meta.get("muscle_group", "General"),
                "difficulty": meta.get("difficulty", "beginner")
            })

            seen_ids.add(ex_id)
            seen_names.add(ex_name_l)

            if len(suggested) >= target_count:
                return suggested

    return suggested


def build_new_schedule_from_split(split_days, user_feedback, search_func, current_plan):
    pain_areas = [str(x).strip() for x in user_feedback.get("pain_areas", [])]
    difficulty = user_feedback.get("difficulty", "")
    liked_exercises = [str(x).strip() for x in user_feedback.get("liked_exercises", [])]
    disliked_exercises = [str(x).strip().lower() for x in user_feedback.get("disliked_exercises", [])]
    blocked_keywords = get_injury_blocked_keywords(pain_areas)

    current_ids = set()
    current_names = set()

    for day in current_plan.get("plan_data", {}).get("schedule", []):
        for ex in day.get("exercises", []):
            if ex.get("exercise_id") is not None:
                current_ids.add(ex["exercise_id"])
            if ex.get("name"):
                current_names.add(str(ex["name"]).strip().lower())

    new_schedule = []
    changes_summary = []

    for item in split_days:
        day_number = item["day"]
        focus = normalize_focus_label(item["focus"])

        queries = build_queries_for_focus_dynamic(
            focus=focus,
            pain_areas=pain_areas,
            difficulty=difficulty,
            liked_exercises=liked_exercises
        )

        suggestions = collect_suggested_exercises(
            focus=focus,
            search_queries=queries,
            search_func=search_func,
            blocked_keywords=blocked_keywords,
            excluded_ids=current_ids,
            excluded_names=current_names,
            target_count=4
        )

        # fallback خاص إذا اليوم مختلط مثل biceps and triceps    
        if len(suggestions) < 4 and focus in ["biceps and triceps", "chest and triceps", "back and biceps", "full body"]:
            parts = []
            if focus == "biceps and triceps":
                parts = ["biceps", "triceps"]
            elif focus == "chest and triceps":
                parts = ["chest", "triceps"]
            elif focus == "back and biceps":
                parts = ["back", "biceps"]
            elif focus == "full body":
                parts = ["chest", "back", "legs", "core"]

            mixed = []
            for part in parts:
                part_queries = build_queries_for_focus_dynamic(
                    focus=part,
                    pain_areas=pain_areas,
                    difficulty=difficulty,
                    liked_exercises=liked_exercises
                )
                part_suggestions = collect_suggested_exercises(
                    focus=part,
                    search_queries=part_queries,
                    search_func=search_func,
                    blocked_keywords=blocked_keywords,
                    excluded_ids=current_ids,
                    excluded_names=current_names,
                    target_count=2 if focus != "full body" else 1
                )
                mixed.extend(part_suggestions)

            deduped = []
            seen_local_ids = set()
            for ex in mixed:
                ex_name = str(ex.get("name", "")).strip()
                if ex["id"] in seen_local_ids:
                    continue
                if not exercise_matches_focus(ex_name, focus):
                    continue
                seen_local_ids.add(ex["id"])
                deduped.append(ex)

            suggestions = deduped[:4]

        day_exercises = []
        for suggestion in suggestions[:4]:
            exercise_data = {
                "exercise_id": suggestion["id"],
                "name": suggestion["name"],
                "muscle_group": suggestion["muscle_group"],
                "difficulty": suggestion["difficulty"],
                "sets": 2 if difficulty == "too_hard" else 3,
                "reps": "12-15" if difficulty == "too_hard" else "8-12",
                "rest_seconds": 90
            }
            day_exercises.append(exercise_data)
            current_ids.add(suggestion["id"])
            current_names.add(suggestion["name"].strip().lower())

        if day_exercises:
            new_schedule.append({
                "day": day_number,
                "focus": focus.title(),
                "exercises": day_exercises
            })
            changes_summary.append(f"Built day {day_number} as {focus.title()} workout")

    new_schedule.sort(key=lambda x: x["day"])
    return new_schedule, changes_summary


def analyze_plan_and_suggest_modifications(current_plan, user_feedback, search_func):
    modified_plan = copy.deepcopy(current_plan)
    changes_summary = []
    recommendations = []

    pain_areas = [str(x).strip() for x in user_feedback.get("pain_areas", [])]
    difficulty = user_feedback.get("difficulty", "")
    disliked_exercises = [str(x).strip().lower() for x in user_feedback.get("disliked_exercises", [])]
    liked_exercises = [str(x).strip() for x in user_feedback.get("liked_exercises", [])]
    modification_request = str(user_feedback.get("modification_request", "")).strip()

    requested_split = extract_requested_split(modification_request)

    if requested_split:
        new_schedule, split_changes = build_new_schedule_from_split(
            split_days=requested_split,
            user_feedback=user_feedback,
            search_func=search_func,
            current_plan=current_plan
        )

        if new_schedule:
            if "plan_data" not in modified_plan:
                modified_plan["plan_data"] = {}

            modified_plan["plan_data"]["schedule"] = new_schedule
            changes_summary.extend(split_changes)

            if pain_areas:
                recommendations.append(
                    f"Avoid exercises that strain the {', '.join(pain_areas)} and use controlled motion."
                )

            if difficulty == "too_hard":
                recommendations.append(
                    "Reduce load, keep reps moderate to high, and avoid painful lockout."
                )
                recommendations.append(
                    "Prefer machine or cable movements when available."
                )
            elif difficulty == "too_easy":
                recommendations.append(
                    "Use slightly more challenging movements or increase training volume gradually."
                )

            if liked_exercises:
                recommendations.append(
                    "Use the preferred exercise variations when they are pain-free and controlled."
                )

            if disliked_exercises:
                recommendations.append(
                    "Avoid disliked movements when safer or more suitable alternatives are available."
                )

            recommendations.append(
                "Stop any movement that causes sharp pain and focus on proper form."
            )

            return {
                "plan_id": current_plan.get("plan_id", "unknown"),
                "version": int(current_plan.get("version", 1)) + 1,
                "changes_summary": changes_summary[:20],
                "modified_plan": modified_plan,
                "recommendations": recommendations[:6]
            }

    return {
        "plan_id": current_plan.get("plan_id", "unknown"),
        "version": int(current_plan.get("version", 1)) + 1,
        "changes_summary": ["No meaningful training split changes detected."],
        "modified_plan": modified_plan,
        "recommendations": ["Keep monitoring form, pain response, and weekly recovery."]
    }

# =====================
# 🚀 FASTAPI APP
# =====================
app = FastAPI(title=settings.APP_NAME, debug=settings.DEBUG)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================
# 📍 APIs
# =====================
@app.get("/")
async def root():
    return {"service": "Gym AI Service", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

def resolve_search_params(query: Optional[str], n_results: Optional[int], body: Optional[SearchRequest]):
    resolved_query = query or (body.query if body else None)
    resolved_n_results = n_results

    if resolved_n_results is None and body:
        resolved_n_results = body.n_results
    if resolved_n_results is None:
        resolved_n_results = 10

    if not resolved_query or not str(resolved_query).strip():
        raise HTTPException(status_code=422, detail="query is required")

    try:
        resolved_n_results = int(resolved_n_results)
    except (TypeError, ValueError):
        resolved_n_results = 10

    if resolved_n_results <= 0:
        resolved_n_results = 10

    return str(resolved_query).strip(), resolved_n_results

def first_present(*values):
    for value in values:
        if value is not None:
            return value
    return None

def resolve_debug_context(debug_context: Optional[bool], body: Optional[SearchRequest]):
    return bool(first_present(debug_context, body.debug_context if body else None, False))

def build_exercise_filters(body, difficulty=None, muscle_group=None, goal=None, exclude_junk=None):
    return {
        "difficulty": first_present(difficulty, body.difficulty if body else None),
        "muscle_group": first_present(muscle_group, body.muscle_group if body else None),
        "goal": first_present(goal, body.goal if body else None),
        "exclude_junk": bool(first_present(exclude_junk, body.exclude_junk if body else None, True)),
    }

def build_food_filters(body, category=None, min_protein=None, max_calories=None, goal=None, exclude_junk=None):
    return {
        "category": first_present(category, body.category if body else None),
        "min_protein": first_present(min_protein, body.min_protein if body else None),
        "max_calories": first_present(max_calories, body.max_calories if body else None),
        "goal": first_present(goal, body.goal if body else None),
        "exclude_junk": bool(first_present(exclude_junk, body.exclude_junk if body else None, True)),
    }

def distance_to_score(distance):
    distance = safe_float(distance, None)
    if distance is None:
        return None
    if distance < 0:
        distance = 0
    return round(1 / (1 + distance), 4)

def preview_text(document, max_len=220):
    text = re.sub(r"\s+", " ", safe_str(document)).strip()
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."

def build_exercise_context(results):
    blocks = []
    for item in unpack_chroma_results(results):
        meta = item.get("metadata") or {}
        blocks.append("\n".join([
            f"Exercise ID: {meta.get('id', item.get('id'))}",
            f"Name: {safe_str(meta.get('name'), 'Exercise')}",
            f"Muscle: {safe_str(meta.get('normalized_muscle_group'), safe_str(meta.get('muscle_group'), 'general'))}",
            f"Body area: {safe_str(meta.get('body_area'), 'general')}",
            f"Difficulty: {safe_str(meta.get('difficulty'), 'unknown')}",
            f"Goal tags: {safe_str(meta.get('goal_tags'), '')}",
            f"Search quality: {item_search_quality(meta)}",
            f"Context: {preview_text(item.get('document'), 500)}",
        ]))
    return "\n\n".join(blocks)

def build_food_context(results):
    blocks = []
    for item in unpack_chroma_results(results):
        meta = item.get("metadata") or {}
        blocks.append("\n".join([
            f"Food ID: {meta.get('id', item.get('id'))}",
            f"Name: {safe_str(meta.get('name'), 'Food')}",
            f"Category: {safe_str(meta.get('normalized_category'), safe_str(meta.get('category'), 'unknown'))}",
            f"Macros: {safe_float(meta.get('calories')):g} cal, {safe_float(meta.get('protein')):g}g protein, {safe_float(meta.get('carbs')):g}g carbs, {safe_float(meta.get('fat')):g}g fat",
            f"Goal tags: {safe_str(meta.get('goal_tags'), '')}",
            f"Meal roles: {safe_str(meta.get('meal_role_tags'), '')}",
            f"Search quality: {item_search_quality(meta)}",
            f"Context: {preview_text(item.get('document'), 500)}",
        ]))
    return "\n\n".join(blocks)

def format_search_response(query: str, result_type: str, results: Dict[str, Any], debug_context: bool = False):
    formatted = []
    source_table = "exercises" if result_type == "exercises" else "foods"

    for raw_item in unpack_chroma_results(results):
        doc = raw_item.get("document")
        meta = raw_item.get("metadata") or {}
        distance = raw_item.get("distance")
        item = {
            "document": doc,
            "metadata": meta,
            "id": raw_item.get("id"),
            "distance": distance,
            "source_id": meta.get("id"),
            "source_table": source_table,
            "source_name": meta.get("name"),
            "score": distance_to_score(distance),
            "preview": preview_text(doc),
            "search_quality": item_search_quality(meta),
        }
        formatted.append(item)

    response = {
        "status": "success",
        "query": query,
        "type": result_type,
        "count": len(formatted),
        "results": formatted,
    }
    if debug_context:
        response["context_preview"] = (
            build_exercise_context(results)
            if result_type == "exercises"
            else build_food_context(results)
        )
    return response

@app.post("/sync-all")
async def sync_all_data(full_sync: bool = Query(False)):
    try:
        result = sync_all(full_sync=full_sync)
        return {
            "status": "success",
            "message": "Sync completed",
            "stats": result,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search-exercises")
async def search_exercises_api(
    body: Optional[SearchRequest] = None,
    query: Optional[str] = Query(None),
    n_results: Optional[int] = Query(None),
    difficulty: Optional[str] = Query(None),
    muscle_group: Optional[str] = Query(None),
    goal: Optional[str] = Query(None),
    exclude_junk: Optional[bool] = Query(None),
    debug_context: Optional[bool] = Query(None),
):
    try:
        resolved_query, resolved_n_results = resolve_search_params(query, n_results, body)
        filters = build_exercise_filters(
            body,
            difficulty=difficulty,
            muscle_group=muscle_group,
            goal=goal,
            exclude_junk=exclude_junk,
        )
        results = search_exercises(resolved_query, resolved_n_results, filters=filters)
        return format_search_response(
            resolved_query,
            "exercises",
            results,
            debug_context=resolve_debug_context(debug_context, body),
        )
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search-foods")
async def search_foods_api(
    body: Optional[SearchRequest] = None,
    query: Optional[str] = Query(None),
    n_results: Optional[int] = Query(None),
    category: Optional[str] = Query(None),
    min_protein: Optional[float] = Query(None),
    max_calories: Optional[float] = Query(None),
    goal: Optional[str] = Query(None),
    exclude_junk: Optional[bool] = Query(None),
    debug_context: Optional[bool] = Query(None),
):
    try:
        resolved_query, resolved_n_results = resolve_search_params(query, n_results, body)
        filters = build_food_filters(
            body,
            category=category,
            min_protein=min_protein,
            max_calories=max_calories,
            goal=goal,
            exclude_junk=exclude_junk,
        )
        results = search_nutrition(resolved_query, resolved_n_results, filters=filters)
        return format_search_response(
            resolved_query,
            "foods",
            results,
            debug_context=resolve_debug_context(debug_context, body),
        )
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def generate_training_plan_rule_based(request: GenerateTrainingRequest):
    try:
        goal = str(request.user_summary.goal or "").strip().lower().replace(" ", "_")
        level = str(request.user_summary.level or "beginner").strip().lower()
        injuries = [str(x).strip().lower() for x in (request.user_summary.injuries or [])]
        weak_points = [str(x).strip().lower() for x in (request.user_summary.weak_points or [])]

        split_days = [
            {"day": 1, "focus": "chest and triceps"},
            {"day": 2, "focus": "back and biceps"},
            {"day": 3, "focus": "shoulders"},
            {"day": 4, "focus": "legs"},
            {"day": 5, "focus": "full body"},
        ]

        focus_keyword_map = {
            "chest and triceps": ["chest", "press", "fly", "pec", "tricep", "pushdown", "extension"],
            "back and biceps": ["back", "row", "pulldown", "pull", "lat", "bicep", "curl"],
            "shoulders": ["shoulder", "lateral raise", "rear delt", "face pull"],
            "legs": ["leg", "squat", "press", "curl", "hamstring", "quad", "calf", "lunge"],
            "core": ["plank", "dead bug", "crunch", "core", "ab", "knee raise", "mountain climber"],
            "full body": ["chest", "row", "leg", "core"],
        }

        injury_block_map = {
            "elbow": [
                "curl", "pushdown", "kickback", "skull crusher", "overhead extension",
                "close grip", "reverse grip", "arnold press", "shoulder press",
                "barbell curl", "ez bar curl", "concentration curl", "hammer curl",
                "cable bicep curl", "dumbbell bicep curl", "tricep", "dip", "push up"
            ],
            "shoulder": [
                "shoulder press", "arnold press", "upright row", "front raise",
                "bench press", "incline press", "pec deck", "fly", "push up"
            ],
            "knee": [
                "leg extension", "walking lunge", "jump squat", "bulgarian",
                "step up", "box step up", "deep squat"
            ],
            "lower back": [
                "romanian deadlift", "deadlift", "good morning", "bent over row"
            ],
            "back": [
                "romanian deadlift", "deadlift", "good morning", "bent over row"
            ],
            "wrist": [
                "barbell curl", "straight bar", "bench press", "push up"
            ],
        }

        blocked_keywords = set()
        for injury in injuries:
            for key, blocked_list in injury_block_map.items():
                if key in injury:
                    blocked_keywords.update(x.lower() for x in blocked_list)

        sets_value = 2 if level == "beginner" else 3
        reps_value = "12-15" if level == "beginner" else "8-12"

        def matches_blocked(exercise_name: str) -> bool:
            name_l = str(exercise_name or "").strip().lower()
            return any(word in name_l for word in blocked_keywords)

        def matches_focus(exercise_name: str, focus: str) -> bool:
            name_l = str(exercise_name or "").strip().lower()
            keywords = focus_keyword_map.get(focus.lower(), [focus.lower()])
            return any(k in name_l for k in keywords)

        def build_queries_for_focus_dynamic(focus: str):
            queries = [f"{level} {focus} exercise"]

            if goal in ["muscle_gain", "higher_protein"]:
                queries.append(f"{focus} hypertrophy exercise")
            elif goal in ["fat_loss", "lower_calories", "weight_loss"]:
                queries.append(f"{focus} fat loss exercise")
            else:
                queries.append(f"{goal} {focus} exercise")

            for point in weak_points[:3]:
                queries.append(f"{focus} exercise for {point}")

            for injury in injuries[:3]:
                queries.append(f"safe {focus} exercise for {injury}")
                queries.append(f"{injury} friendly {focus} exercise")

            deduped = []
            seen = set()
            for q in queries:
                q = q.strip().lower()
                if q and q not in seen:
                    seen.add(q)
                    deduped.append(q)
            return deduped[:10]

        def make_exercise_item(meta):
            return {
                "exercise_id": meta["id"],
                "name": meta["name"],
                "muscle_group": meta.get("muscle_group", "General"),
                "difficulty": meta.get("difficulty", level or "beginner"),
                "sets": sets_value,
                "reps": reps_value,
                "rest_seconds": 90
            }

        def search_and_collect(focus: str, used_ids: set, used_names: set, target_count: int = 4):
            collected = []
            queries = build_queries_for_focus_dynamic(focus)

            for query in queries:
                results = search_exercises(query, n_results=20)
                docs = results.get("documents", [[]])
                metas = results.get("metadatas", [[]])

                if not docs or not metas:
                    continue

                for _, meta in zip(docs[0], metas[0]):
                    ex_id = meta.get("id")
                    ex_name = str(meta.get("name", "")).strip()
                    ex_name_l = ex_name.lower()

                    if not ex_id or not ex_name:
                        continue
                    if ex_id in used_ids or ex_name_l in used_names:
                        continue
                    if matches_blocked(ex_name):
                        continue
                    if not matches_focus(ex_name, focus):
                        continue

                    collected.append(make_exercise_item(meta))
                    used_ids.add(ex_id)
                    used_names.add(ex_name_l)

                    if len(collected) >= target_count:
                        return collected

            fallback_queries = [
                f"{focus} workout exercise",
                f"beginner {focus} exercise",
                f"safe {focus} exercise"
            ]

            for query in fallback_queries:
                results = search_exercises(query, n_results=25)
                docs = results.get("documents", [[]])
                metas = results.get("metadatas", [[]])

                if not docs or not metas:
                    continue

                for _, meta in zip(docs[0], metas[0]):
                    ex_id = meta.get("id")
                    ex_name = str(meta.get("name", "")).strip()
                    ex_name_l = ex_name.lower()

                    if not ex_id or not ex_name:
                        continue
                    if ex_id in used_ids or ex_name_l in used_names:
                        continue
                    if matches_blocked(ex_name):
                        continue
                    if not matches_focus(ex_name, focus):
                        continue

                    collected.append(make_exercise_item(meta))
                    used_ids.add(ex_id)
                    used_names.add(ex_name_l)

                    if len(collected) >= target_count:
                        return collected

            return collected

        def collect_full_body_exercises(used_ids: set, used_names: set):
            collected = []
            full_body_parts = ["chest and triceps", "back and biceps", "legs", "core"]

            for part in full_body_parts:
                exercises = search_and_collect(part, used_ids, used_names, target_count=1)
                if exercises:
                    collected.extend(exercises[:1])

            if len(collected) < 4:
                fallback_queries = [
                    "beginner full body exercise",
                    "safe full body workout exercise",
                    "core exercise"
                ]

                for query in fallback_queries:
                    results = search_exercises(query, n_results=20)
                    docs = results.get("documents", [[]])
                    metas = results.get("metadatas", [[]])

                    if not docs or not metas:
                        continue

                    for _, meta in zip(docs[0], metas[0]):
                        ex_id = meta.get("id")
                        ex_name = str(meta.get("name", "")).strip()
                        ex_name_l = ex_name.lower()

                        if not ex_id or not ex_name:
                            continue
                        if ex_id in used_ids or ex_name_l in used_names:
                            continue
                        if matches_blocked(ex_name):
                            continue

                        collected.append(make_exercise_item(meta))
                        used_ids.add(ex_id)
                        used_names.add(ex_name_l)

                        if len(collected) >= 4:
                            return collected

            return collected[:4]

        used_ids = set()
        used_names = set()
        schedule = []

        for split in split_days:
            focus = split["focus"]

            if focus == "full body":
                day_exercises = collect_full_body_exercises(used_ids, used_names)
            else:
                day_exercises = search_and_collect(
                    focus=focus,
                    used_ids=used_ids,
                    used_names=used_names,
                    target_count=4
                )

            if day_exercises:
                schedule.append({
                    "day": split["day"],
                    "focus": focus.title(),
                    "exercises": day_exercises
                })

        return {
            "plan_id": f"plan_{request.user_summary.user_id}_{int(datetime.now().timestamp())}",
            "version": 1,
            "generated_at": datetime.now().isoformat(),
            "plan_data": {
                "duration_weeks": 4,
                "schedule": schedule
            }
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))  

def model_to_plain_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model or {})

def first_training_preference(preferences: Dict[str, Any], keys: List[str]):
    for key in keys:
        value = preferences.get(key)
        if value not in [None, "", []]:
            return value
    return None

def build_training_rag_search_query(request: GenerateTrainingRequest):
    user = request.user_summary
    preferences = request.preferences or {}
    pieces = [
        str(user.level or "beginner"),
        str(user.goal or "fitness"),
        "training plan exercises",
    ]

    days_per_week = first_training_preference(
        preferences,
        ["days_per_week", "training_days_per_week", "available_days", "workout_days"],
    )
    if days_per_week:
        pieces.append(f"{days_per_week} days per week")

    split = first_training_preference(preferences, ["preferred_split", "split", "training_split"])
    if split:
        pieces.append(str(split))

    for weak_point in (user.weak_points or [])[:3]:
        pieces.append(f"weak point {weak_point}")

    for injury in (user.injuries or [])[:3]:
        pieces.append(f"safe exercise for {injury}")

    for liked in (preferences.get("liked_exercises", []) or [])[:3]:
        pieces.append(str(liked))

    return " ".join(str(piece).strip() for piece in pieces if str(piece).strip())

def build_training_user_payload(request: GenerateTrainingRequest, retrieval_query: str):
    preferences = request.preferences or {}
    return {
        "user_summary": model_to_plain_dict(request.user_summary),
        "preferences": preferences,
        "previous_plans": request.previous_plans or [],
        "retrieval_query": retrieval_query,
        "days_per_week": first_training_preference(
            preferences,
            ["days_per_week", "training_days_per_week", "available_days", "workout_days"],
        ),
    }

class RAGGenerationError(RuntimeError):
    def __init__(self, category: str, message: str, original_error: Exception = None):
        super().__init__(message)
        self.category = category
        self.original_error = original_error

def sanitize_rag_error_message(error: Exception, max_len: int = 240):
    message = str(error or "")
    if settings.OPENROUTER_API_KEY:
        message = message.replace(settings.OPENROUTER_API_KEY, "[redacted]")
    if settings.GEMINI_API_KEY:
        message = message.replace(settings.GEMINI_API_KEY, "[redacted]")
    message = re.sub(r"\s+", " ", message).strip()
    if len(message) > max_len:
        return message[:max_len].rstrip() + "..."
    return message

def log_rag_exception(stage: str, error: Exception):
    rag_logger.warning(
        "training_rag.%s failed: %s: %s",
        stage,
        type(error).__name__,
        sanitize_rag_error_message(error),
    )

def classify_llm_error(error: Exception):
    message = str(error or "")
    if "OPENROUTER_API_KEY" in message or "OPENROUTER_MODEL" in message:
        return "openrouter_not_configured"
    if "GEMINI_API_KEY" in message or "GEMINI_GENERATION_MODEL" in message:
        return "gemini_generation_failed"
    if "timed out" in message.lower() or "timeout" in type(error).__name__.lower():
        return "gemini_generation_failed" if "Gemini" in message else "openrouter_timeout"
    if "status " in message and "OpenRouter request failed" in message:
        return "openrouter_http_error"
    if "Gemini request failed" in message or "Gemini returned" in message:
        if "empty assistant text" in message or "did not contain assistant text" in message:
            return "llm_empty_response"
        return "gemini_generation_failed"
    if "Unsupported LLM_PROVIDER" in message:
        return "llm_generation_failed"
    if "empty assistant message content" in message or "did not contain assistant message content" in message:
        return "llm_empty_response"
    return "llm_generation_failed"

def safe_training_fallback_reason(error: Exception):
    if isinstance(error, RAGGenerationError):
        return error.category

    message = str(error or "")
    if "OPENROUTER_API_KEY" in message or "OPENROUTER_MODEL" in message:
        return "openrouter_not_configured"
    if "GEMINI_API_KEY" in message or "GEMINI_GENERATION_MODEL" in message:
        return "gemini_generation_failed"
    if "timed out" in message.lower():
        return "gemini_generation_failed" if "Gemini" in message else "openrouter_timeout"
    if "OpenRouter request failed with status" in message:
        return "openrouter_http_error"
    if "Gemini request failed" in message or "Gemini returned" in message:
        if "empty assistant text" in message or "did not contain assistant text" in message:
            return "llm_empty_response"
        return "gemini_generation_failed"
    if "Unsupported LLM_PROVIDER" in message:
        return "llm_generation_failed"
    if "empty assistant message content" in message or "did not contain assistant message content" in message:
        return "llm_empty_response"
    if "No retrieved exercises" in message:
        return "no_retrieved_exercises"
    if "No valid JSON object" in message or "No response text" in message or "JSON" in message:
        return "invalid_llm_json"
    if "validation" in message.lower():
        return "rag_schema_validation_failed"
    return "rag_generation_failed"

def rag_debug_error_type(error: Exception):
    if isinstance(error, RAGGenerationError):
        return error.category
    return safe_training_fallback_reason(error)

def training_rag_response_to_legacy_response(rag_response, request: GenerateTrainingRequest, retrieved_items=None):
    schedule = []
    for day in rag_response.days:
        exercises = []
        for exercise in day.exercises:
            exercises.append({
                "exercise_id": exercise.exercise_id,
                "name": exercise.name,
                "sets": exercise.sets,
                "reps": exercise.reps,
                "rest_seconds": exercise.rest_seconds if exercise.rest_seconds is not None else 90,
                "intensity": exercise.intensity,
                "reason": exercise.reason,
                "source_id": exercise.source_id,
            })

        schedule.append({
            "day": day.day,
            "focus": day.focus,
            "exercises": exercises,
        })

    if not schedule:
        raise ValueError("LLM response did not include training days")
    if not any(day["exercises"] for day in schedule):
        raise ValueError("LLM response did not include usable exercises")

    guard_changes, guard_warnings, _, guard_sources = (
        apply_shoulder_safety_guard_to_modified_schedule(
            schedule,
            request,
            {},
            retrieved_items or [],
        )
    )
    injury_warnings = unique_strings(
        guard_warnings,
        rag_response.injury_warnings,
    )
    used_source_ids = used_training_source_ids(schedule)
    sources = [
        model_to_plain_dict(source)
        for source in rag_response.sources
        if str(source.source_id) in used_source_ids
    ]
    existing_source_ids = {str(source.get("source_id")) for source in sources}
    for source in guard_sources:
        if str(source.get("source_id")) not in existing_source_ids:
            sources.append(source)
            existing_source_ids.add(str(source.get("source_id")))

    return {
        "status": rag_response.status,
        "plan_id": f"plan_{request.user_summary.user_id}_{int(datetime.now().timestamp())}",
        "version": 1,
        "generated_at": datetime.now().isoformat(),
        "plan_data": {
            "duration_weeks": 4,
            "schedule": schedule,
        },
        "generation_mode": "rag",
        "rag_summary": rag_response.summary,
        "sources": sources,
        "injury_warnings": injury_warnings,
    }

def training_plan_id_from_request(request: SmartModifyTrainingRequest):
    return request.current_plan_id or request.current_plan.get("plan_id", "unknown")

def next_training_plan_version(current_plan: Dict[str, Any]):
    try:
        return int(current_plan.get("version", 1)) + 1
    except (TypeError, ValueError):
        return 2

def normalize_training_modification_text(value):
    if value in [None, "", []]:
        return ""
    if isinstance(value, dict):
        parts = []
        for key in ["reason", "notes", "modification_request", "request", "message", "difficulty"]:
            item = value.get(key)
            if item not in [None, "", []]:
                parts.append(f"{key}: {item}")
        if parts:
            return "; ".join(parts)
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value).strip()

def resolve_training_modification_feedback(request: SmartModifyTrainingRequest):
    feedback = model_to_plain_dict(request.user_feedback) if request.user_feedback else {}
    root_modification = request.modification_request

    if isinstance(root_modification, dict):
        for key, value in root_modification.items():
            if key not in feedback or feedback.get(key) in [None, "", []]:
                feedback[key] = value

    root_modification_text = normalize_training_modification_text(root_modification)
    if root_modification_text and not feedback.get("modification_request"):
        feedback["modification_request"] = root_modification_text

    if not feedback.get("modification_request"):
        inline_text = normalize_training_modification_text({
            "reason": feedback.get("reason"),
            "notes": feedback.get("notes"),
            "request": feedback.get("request"),
            "message": feedback.get("message"),
        })
        if inline_text:
            feedback["modification_request"] = inline_text

    if request.user_summary.injuries and not feedback.get("pain_areas"):
        feedback["pain_areas"] = request.user_summary.injuries

    return feedback

def extract_training_schedule(current_plan: Dict[str, Any]):
    if not isinstance(current_plan, dict):
        return []
    plan_data = current_plan.get("plan_data")
    if isinstance(plan_data, dict) and isinstance(plan_data.get("schedule"), list):
        return plan_data.get("schedule") or []
    if isinstance(current_plan.get("schedule"), list):
        return current_plan.get("schedule") or []
    if isinstance(current_plan.get("days"), list):
        return current_plan.get("days") or []
    return []

def build_modify_training_rag_search_query(request: SmartModifyTrainingRequest, user_feedback: Dict[str, Any]):
    user = request.user_summary
    pieces = [
        str(user.level or "beginner"),
        str(user.goal or "fitness"),
        "training plan modification exercises",
        normalize_training_modification_text(user_feedback.get("modification_request")),
    ]

    for injury in (user.injuries or [])[:4]:
        pieces.append(f"safe exercise for {injury}")
    pain_areas = user_feedback.get("pain_areas", [])
    if pain_areas in [None, ""]:
        pain_areas = []
    elif not isinstance(pain_areas, list):
        pain_areas = [pain_areas]
    for pain_area in pain_areas[:4]:
        pieces.append(f"safe exercise for {pain_area}")
    for weak_point in (user.weak_points or [])[:4]:
        pieces.append(f"weak point {weak_point}")

    current_schedule = extract_training_schedule(request.current_plan)
    for day in current_schedule[:4]:
        if not isinstance(day, dict):
            continue
        if day.get("focus"):
            pieces.append(str(day.get("focus")))
        for exercise in (day.get("exercises") or [])[:3]:
            if isinstance(exercise, dict) and exercise.get("name"):
                pieces.append(str(exercise.get("name")))

    return " ".join(str(piece).strip() for piece in pieces if str(piece).strip())

def build_modify_training_user_payload(
    request: SmartModifyTrainingRequest,
    user_feedback: Dict[str, Any],
    retrieval_query: str,
):
    return {
        "current_plan_id": training_plan_id_from_request(request),
        "user_summary": model_to_plain_dict(request.user_summary),
        "user_feedback": user_feedback,
        "modification_request": normalize_training_modification_text(
            user_feedback.get("modification_request")
        ),
        "current_plan": request.current_plan,
        "current_schedule": extract_training_schedule(request.current_plan),
        "retrieval_query": retrieval_query,
    }

def has_shoulder_pain_context(request: SmartModifyTrainingRequest, user_feedback: Dict[str, Any]):
    parts = []
    parts.extend(request.user_summary.injuries or [])

    for key in ["pain_areas", "injuries", "modification_request", "reason", "notes", "request", "message"]:
        value = user_feedback.get(key)
        if isinstance(value, list):
            parts.extend(value)
        elif value not in [None, "", []]:
            parts.append(value)

    text = normalize_term(" ".join(str(part) for part in parts))
    return "shoulder" in text and any(
        marker in text
        for marker in ["pain", "injury", "injuries", "injured", "ache", "discomfort", "sore", "strain", "shoulder"]
    )

def is_rehab_safe_shoulder_context(exercise: Dict[str, Any]):
    text = normalize_term(
        " ".join(
            str(exercise.get(key, ""))
            for key in ["name", "reason", "intensity", "notes", "description"]
        )
    )
    rehab_markers = [
        "rehab",
        "physio",
        "physical therapy",
        "very light",
        "pain free",
        "pain-free",
        "mobility",
        "isometric",
        "external rotation",
        "scapular",
        "band pull apart",
    ]
    return any(marker in text for marker in rehab_markers)

def is_risky_for_shoulder_pain(exercise: Dict[str, Any]):
    name = normalize_term(exercise.get("name", ""))
    if not name:
        return False

    always_blocked = [
        "shoulder press",
        "arnold press",
        "overhead press",
        "military press",
        "upright row",
    ]
    direct_isolation = [
        "lateral raise",
        "front raise",
        "rear delt",
        "face pull",
        "shoulder raise",
        "deltoid raise",
        "delt raise",
    ]
    controlled_press_markers = [
        "machine",
        "cable",
        "light",
        "neutral grip",
        "assisted",
        "supported",
    ]

    if any(keyword in name for keyword in always_blocked):
        return True
    if any(keyword in name for keyword in direct_isolation):
        return True
    if "dip" in name or "pushup" in name or "push up" in name or "push-up" in name:
        return True
    if "bench press" in name:
        return True
    if "incline press" in name or "decline press" in name:
        return True
    if "incline dumbbell press" in name or "dumbbell press" in name:
        return True
    if "heavy" in name and "press" in name:
        return True
    if "barbell" in name and "press" in name and "leg press" not in name:
        return True
    if "press" in name and "leg press" not in name and any(
        marker in name for marker in ["bench", "incline", "decline", "dumbbell", "barbell"]
    ):
        return True
    if "chest press" in name and not any(marker in name for marker in controlled_press_markers):
        return True
    if "fly" in name and not any(marker in name for marker in ["machine", "cable", "light"]):
        return True
    return False

def retrieved_exercise_to_guard_candidate(item):
    meta = item.get("metadata") or {}
    source_id = meta.get("id")
    if source_id in [None, ""]:
        source_id = item.get("id")
    name = safe_str(meta.get("name"), "").strip()
    if not name or source_id in [None, ""]:
        return None
    return {
        "exercise_id": source_id,
        "source_id": source_id,
        "name": name,
        "muscle_group": meta.get("muscle_group"),
        "normalized_muscle_group": meta.get("normalized_muscle_group"),
        "body_area": meta.get("body_area"),
        "goal_tags": meta.get("goal_tags"),
        "search_quality": meta.get("search_quality"),
        "document": item.get("document"),
        "difficulty": meta.get("difficulty"),
        "score": distance_to_score(item.get("distance")),
    }

def guard_candidate_text(candidate):
    return normalize_term(
        " ".join(
            str(candidate.get(key, ""))
            for key in [
                "name",
                "muscle_group",
                "normalized_muscle_group",
                "body_area",
                "goal_tags",
                "difficulty",
                "document",
            ]
        )
    )

def guard_candidate_muscle_group(candidate):
    normalized = normalize_label(candidate.get("normalized_muscle_group"))
    if normalized and normalized not in {"general", "unknown"}:
        return normalized

    inferred = infer_normalized_muscle_group(
        candidate.get("muscle_group"),
        candidate.get("name"),
        description=candidate.get("document"),
    )
    return normalize_label(inferred) or "general"

def guard_candidate_body_area(candidate):
    body_area = normalize_label(candidate.get("body_area"))
    if body_area and body_area not in {"general", "unknown"}:
        return body_area
    return infer_body_area(guard_candidate_muscle_group(candidate), candidate.get("muscle_group"))

def is_lower_body_candidate(candidate):
    text = normalize_term(
        " ".join(
            str(candidate.get(key, ""))
            for key in ["name", "muscle_group", "normalized_muscle_group"]
        )
    )
    lower_markers = [
        "leg",
        "legs",
        "quad",
        "hamstring",
        "calf",
        "glute",
        "squat",
        "lunge",
        "leg press",
        "leg extension",
        "leg curl",
    ]
    return any(marker in text for marker in lower_markers)

def focus_is_upper_or_push(focus):
    text = normalize_term(focus)
    return any(marker in text for marker in ["push", "upper", "chest", "shoulder", "tricep"])

def shoulder_guard_focus_family(focus):
    text = normalize_term(focus)
    if any(marker in text for marker in ["push", "chest", "pec", "tricep"]):
        return "push_chest"
    if any(marker in text for marker in ["pull", "back", "bicep"]):
        return "pull"
    if any(marker in text for marker in ["leg", "lower", "quad", "hamstring", "glute"]):
        return "lower"
    return "general"

def is_controlled_upper_candidate(candidate):
    text = guard_candidate_text(candidate)
    return any(
        marker in text
        for marker in [
            "machine",
            "cable",
            "rope",
            "band",
            "light",
            "supported",
            "seated",
            "neutral grip",
            "assisted",
            "pec deck",
        ]
    )

def is_chest_guard_candidate(candidate):
    text = guard_candidate_text(candidate)
    muscle = guard_candidate_muscle_group(candidate)
    if "chest supported" in text and "row" in text:
        return False
    return muscle == "chest" or any(
        marker in text
        for marker in [
            "chest press",
            "pec deck",
            "cable crossover",
            "cable fly",
            "chest fly",
            "chest machine",
        ]
    )

def is_safe_controlled_chest_candidate(candidate):
    return is_chest_guard_candidate(candidate) and is_controlled_upper_candidate(candidate)

def is_triceps_isolation_candidate(candidate):
    text = guard_candidate_text(candidate)
    muscle = guard_candidate_muscle_group(candidate)
    return muscle == "triceps" or any(
        marker in text
        for marker in [
            "tricep pushdown",
            "triceps pushdown",
            "rope pushdown",
            "tricep extension",
            "triceps extension",
            "kickback",
        ]
    )

def is_pull_or_arm_pull_candidate(candidate):
    text = guard_candidate_text(candidate)
    muscle = guard_candidate_muscle_group(candidate)
    if muscle in {"back", "biceps"}:
        return True
    return any(
        marker in text
        for marker in [" row", "pulldown", "pull down", "pullup", "pull up", "lat ", "bicep", "curl"]
    )

def is_neutral_safe_upper_candidate(candidate):
    if is_lower_body_candidate(candidate) or is_pull_or_arm_pull_candidate(candidate):
        return False
    muscle = guard_candidate_muscle_group(candidate)
    body_area = guard_candidate_body_area(candidate)
    if is_rehab_safe_shoulder_context(candidate):
        return True
    if muscle in {"arms", "upper_body", "general"} and body_area in {"upper_body", "general"}:
        return True
    return False

def shoulder_guard_push_candidate_tier(candidate):
    if is_lower_body_candidate(candidate) or is_pull_or_arm_pull_candidate(candidate):
        return None
    if is_safe_controlled_chest_candidate(candidate):
        return 0
    if is_triceps_isolation_candidate(candidate):
        return 1
    if is_chest_guard_candidate(candidate):
        return 2
    if is_neutral_safe_upper_candidate(candidate):
        return 3
    return None

def shoulder_guard_focus_alignment_score(candidate, focus):
    family = shoulder_guard_focus_family(focus)
    muscle = guard_candidate_muscle_group(candidate)
    score = 0

    if family == "push_chest":
        if is_safe_controlled_chest_candidate(candidate):
            score += 5
        elif is_chest_guard_candidate(candidate):
            score += 4
        if is_triceps_isolation_candidate(candidate):
            score += 4
        if is_neutral_safe_upper_candidate(candidate):
            score += 1
        return score

    focus_text = normalize_term(focus)
    if muscle and muscle in focus_text:
        score += 4
    if exercise_matches_focus(candidate.get("name"), focus):
        score += 3
    if family == "pull" and muscle in {"back", "biceps"}:
        score += 4
    if family == "lower" and is_lower_body_candidate(candidate):
        score += 4
    return score

def shoulder_guard_candidate_rank(candidate, focus=None):
    if is_risky_for_shoulder_pain(candidate):
        return None

    family = shoulder_guard_focus_family(focus)
    if family == "push_chest":
        tier = shoulder_guard_push_candidate_tier(candidate)
        if tier is None:
            return None
    else:
        if focus_is_upper_or_push(focus) and is_lower_body_candidate(candidate):
            return None
        tier = 0 if shoulder_guard_focus_alignment_score(candidate, focus) > 0 else 2

    retrieved_score = safe_float(candidate.get("score"), 0.0) or 0.0
    return (
        tier,
        -shoulder_guard_focus_alignment_score(candidate, focus),
        -int(is_controlled_upper_candidate(candidate)),
        -retrieved_score,
        safe_str(candidate.get("name")).lower(),
    )

def find_shoulder_safe_retrieved_replacement(retrieved_items, used_source_ids, focus=None):
    ranked_candidates = []
    for item in retrieved_items or []:
        candidate = retrieved_exercise_to_guard_candidate(item)
        if not candidate:
            continue
        candidate_id = str(candidate.get("source_id"))
        if candidate_id in used_source_ids:
            continue
        if is_risky_for_shoulder_pain(candidate):
            continue
        rank = shoulder_guard_candidate_rank(candidate, focus=focus)
        if rank is None:
            continue
        ranked_candidates.append((rank, candidate))

    if not ranked_candidates:
        return None

    ranked_candidates.sort(key=lambda item: item[0])
    return ranked_candidates[0][1]

def apply_shoulder_safety_guard_to_modified_schedule(
    schedule,
    request: SmartModifyTrainingRequest,
    user_feedback: Dict[str, Any],
    retrieved_items,
):
    if not has_shoulder_pain_context(request, user_feedback):
        return [], [], [], []

    guard_changes = []
    guard_warnings = [
        "Shoulder pain reported: avoid overhead pressing and direct shoulder-isolation movements unless cleared and pain-free.",
        "If shoulder pain persists or worsens, stop the exercise and review the plan with a coach or physiotherapist.",
    ]
    guard_recommendations = [
        "Use pain-free ranges of motion, conservative loads, and avoid forcing shoulder-focused replacements."
    ]
    guard_sources = []
    used_source_ids = {
        str(exercise.get("source_id") or exercise.get("exercise_id"))
        for day in schedule
        for exercise in day.get("exercises", [])
        if exercise.get("source_id") is not None or exercise.get("exercise_id") is not None
    }

    for day in schedule:
        safe_exercises = []
        for exercise in day.get("exercises", []):
            if not is_risky_for_shoulder_pain(exercise):
                safe_exercises.append(exercise)
                continue

            risky_name = safe_str(exercise.get("name"), "risky shoulder exercise")
            replacement = find_shoulder_safe_retrieved_replacement(
                retrieved_items,
                used_source_ids,
                focus=day.get("focus"),
            )
            if replacement:
                replacement_exercise = {
                    "exercise_id": replacement["exercise_id"],
                    "name": replacement["name"],
                    "sets": exercise.get("sets", 2),
                    "reps": exercise.get("reps", "10-12"),
                    "rest_seconds": exercise.get("rest_seconds", 90),
                    "intensity": "light_to_moderate",
                    "reason": (
                        f"Safety guard replacement for {risky_name}: shoulder pain was reported, "
                        "so direct shoulder/overhead stress was avoided."
                    ),
                    "source_id": replacement["source_id"],
                }
                safe_exercises.append(replacement_exercise)
                used_source_ids.add(str(replacement["source_id"]))
                guard_sources.append({
                    "source_id": replacement["source_id"],
                    "source_table": "exercises",
                    "source_name": replacement["name"],
                    "score": replacement.get("score"),
                    "reason_used": "Used by shoulder safety guard as a safer retrieved replacement.",
                })
                guard_changes.append(
                    f"Safety guard replaced '{risky_name}' with '{replacement['name']}' because shoulder pain was reported."
                )
            else:
                removal_message = (
                    f"Safety guard removed '{risky_name}' because shoulder pain was reported "
                    "and no safer focus-aligned retrieved replacement was available."
                )
                guard_changes.append(removal_message)
                guard_warnings.append(removal_message)

        day["exercises"] = safe_exercises

    return guard_changes, guard_warnings, guard_recommendations, guard_sources

def unique_strings(*groups, limit=None):
    merged = []
    for group in groups:
        for value in group or []:
            if value not in merged:
                merged.append(value)
            if limit and len(merged) >= limit:
                return merged
    return merged

def used_training_source_ids(schedule):
    return {
        str(exercise.get("source_id") or exercise.get("exercise_id"))
        for day in schedule
        for exercise in day.get("exercises", [])
        if exercise.get("source_id") is not None or exercise.get("exercise_id") is not None
    }

def merge_sources_for_used_training_exercises(existing_sources, guard_sources, schedule):
    used_source_ids = used_training_source_ids(schedule)
    merged = []
    seen = set()

    for source in (existing_sources or []) + (guard_sources or []):
        source_dict = model_to_plain_dict(source)
        source_id = str(source_dict.get("source_id"))
        if source_id not in used_source_ids or source_id in seen:
            continue
        merged.append(source_dict)
        seen.add(source_id)

    return merged

def fetch_retrieved_items_for_modify_training_guard(
    request: SmartModifyTrainingRequest,
    user_feedback: Dict[str, Any],
):
    try:
        retrieval_query = build_modify_training_rag_search_query(request, user_feedback)
        results = search_exercises(retrieval_query, n_results=30)
        return unpack_chroma_results(results)
    except Exception as error:
        log_rag_exception("modify_training_guard_retrieval", error)
        return []

def apply_shoulder_safety_guard_to_legacy_modify_result(
    result,
    request: SmartModifyTrainingRequest,
    user_feedback: Dict[str, Any],
):
    if not isinstance(result, dict):
        return result

    modified_plan = result.get("modified_plan")
    if not isinstance(modified_plan, dict):
        return result

    schedule = extract_training_schedule(modified_plan)
    if not schedule:
        return result

    retrieved_items = fetch_retrieved_items_for_modify_training_guard(request, user_feedback)
    guard_changes, guard_warnings, guard_recommendations, guard_sources = (
        apply_shoulder_safety_guard_to_modified_schedule(
            schedule,
            request,
            user_feedback,
            retrieved_items,
        )
    )
    if not guard_changes and not guard_warnings and not guard_recommendations:
        return result

    plan_data = modified_plan.get("plan_data")
    if not isinstance(plan_data, dict):
        plan_data = {}
    plan_data["schedule"] = schedule
    modified_plan["plan_data"] = plan_data
    if "schedule" in modified_plan:
        modified_plan["schedule"] = schedule

    result["changes_summary"] = unique_strings(
        guard_changes,
        result.get("changes_summary"),
        limit=20,
    )
    result["recommendations"] = unique_strings(
        result.get("recommendations"),
        guard_recommendations,
        limit=8,
    )
    result["injury_warnings"] = unique_strings(
        guard_warnings,
        result.get("injury_warnings"),
    )
    result["sources"] = merge_sources_for_used_training_exercises(
        result.get("sources"),
        guard_sources,
        schedule,
    )
    return result

def modified_training_rag_response_to_legacy_response(
    rag_response,
    request: SmartModifyTrainingRequest,
    user_feedback: Dict[str, Any],
    retrieved_items=None,
):
    schedule = []
    for day in rag_response.modified_plan.schedule:
        exercises = []
        for exercise in day.exercises:
            exercises.append({
                "exercise_id": exercise.exercise_id,
                "name": exercise.name,
                "sets": exercise.sets,
                "reps": exercise.reps,
                "rest_seconds": exercise.rest_seconds if exercise.rest_seconds is not None else 90,
                "intensity": exercise.intensity,
                "reason": exercise.reason,
                "source_id": exercise.source_id,
            })

        schedule.append({
            "day": day.day,
            "focus": day.focus,
            "exercises": exercises,
        })

    if not schedule:
        raise ValueError("LLM response did not include a modified training schedule")

    guard_changes, guard_warnings, guard_recommendations, guard_sources = (
        apply_shoulder_safety_guard_to_modified_schedule(
            schedule,
            request,
            user_feedback,
            retrieved_items or [],
        )
    )

    modified_plan = copy.deepcopy(request.current_plan or {})
    if not isinstance(modified_plan, dict):
        modified_plan = {}

    plan_data = modified_plan.get("plan_data")
    if not isinstance(plan_data, dict):
        plan_data = {}
    plan_data["schedule"] = schedule
    modified_plan["plan_data"] = plan_data

    if "schedule" in modified_plan or "plan_data" not in (request.current_plan or {}):
        modified_plan["schedule"] = schedule

    changes_summary = list(guard_changes)
    changes_summary.extend(rag_response.changes_summary or [rag_response.summary])
    recommendations = list(rag_response.recommendations or [
        "Stop any movement that causes sharp pain and adjust load conservatively."
    ])
    recommendations.extend(guard_recommendations)
    injury_warnings = list(guard_warnings)
    injury_warnings.extend(
        warning for warning in (rag_response.injury_warnings or []) if warning not in injury_warnings
    )
    used_source_ids = used_training_source_ids(schedule)
    sources = [
        model_to_plain_dict(source)
        for source in rag_response.sources
        if str(source.source_id) in used_source_ids
    ]
    existing_source_ids = {str(source.get("source_id")) for source in sources}
    for source in guard_sources:
        if str(source.get("source_id")) not in existing_source_ids:
            sources.append(source)
            existing_source_ids.add(str(source.get("source_id")))

    return {
        "status": rag_response.status,
        "plan_id": training_plan_id_from_request(request),
        "version": next_training_plan_version(request.current_plan),
        "changes_summary": changes_summary[:20],
        "modified_plan": modified_plan,
        "recommendations": recommendations[:8],
        "generation_mode": "rag",
        "rag_summary": rag_response.summary,
        "sources": sources,
        "injury_warnings": injury_warnings,
    }

def generate_training_plan_rag(request: GenerateTrainingRequest):
    retrieval_query = build_training_rag_search_query(request)
    rag_logger.info("training_rag.retrieval_start user_id=%s", request.user_summary.user_id)

    try:
        results = search_exercises(retrieval_query, n_results=30)
        retrieved_items = unpack_chroma_results(results)
    except Exception as error:
        log_rag_exception("retrieval", error)
        raise RAGGenerationError("rag_generation_failed", sanitize_rag_error_message(error), error) from error

    rag_logger.info("training_rag.retrieval_done retrieved_exercises=%s", len(retrieved_items))
    if not retrieved_items:
        message = "No retrieved exercises found for RAG training generation"
        rag_logger.warning("training_rag.retrieval_empty: %s", message)
        raise RAGGenerationError("no_retrieved_exercises", message)

    exercise_context = build_exercise_context(results)
    if not exercise_context.strip():
        message = "No retrieved exercises found for RAG training generation"
        rag_logger.warning("training_rag.context_empty: %s", message)
        raise RAGGenerationError("no_retrieved_exercises", message)

    user_payload = build_training_user_payload(request, retrieval_query)
    messages = build_training_plan_prompt(user_payload, exercise_context)
    rag_logger.info(
        "training_rag.prompt_built message_count=%s context_chars=%s",
        len(messages),
        len(exercise_context),
    )

    try:
        rag_logger.info("training_rag.llm_call_start provider=%s", settings.LLM_PROVIDER)
        llm_text = call_llm_chat(messages, response_schema=training_plan_response_schema())
        rag_logger.info("training_rag.llm_call_done provider=%s response_chars=%s", settings.LLM_PROVIDER, len(llm_text or ""))
    except Exception as error:
        log_rag_exception("llm_call", error)
        category = classify_llm_error(error)
        raise RAGGenerationError(category, sanitize_rag_error_message(error), error) from error

    try:
        rag_response = parse_training_plan_response(llm_text)
        rag_logger.info("training_rag.parse_done")
    except ValidationError as error:
        log_rag_exception("parse_validation", error)
        raise RAGGenerationError(
            "rag_schema_validation_failed",
            sanitize_rag_error_message(error),
            error,
        ) from error
    except ValueError as error:
        log_rag_exception("parse_json", error)
        raise RAGGenerationError("invalid_llm_json", sanitize_rag_error_message(error), error) from error
    except Exception as error:
        log_rag_exception("parse_unknown", error)
        raise RAGGenerationError("invalid_llm_json", sanitize_rag_error_message(error), error) from error

    try:
        response = training_rag_response_to_legacy_response(
            rag_response,
            request,
            retrieved_items=retrieved_items,
        )
        rag_logger.info("training_rag.legacy_conversion_done")
        return response
    except Exception as error:
        log_rag_exception("legacy_conversion", error)
        raise RAGGenerationError(
            "rag_legacy_conversion_failed",
            sanitize_rag_error_message(error),
            error,
        ) from error

def generate_modified_training_plan_rag(request: SmartModifyTrainingRequest):
    user_feedback = resolve_training_modification_feedback(request)
    retrieval_query = build_modify_training_rag_search_query(request, user_feedback)
    rag_logger.info("modify_training_rag.retrieval_start user_id=%s", request.user_summary.user_id)

    try:
        results = search_exercises(retrieval_query, n_results=30)
        retrieved_items = unpack_chroma_results(results)
    except Exception as error:
        log_rag_exception("modify_training_retrieval", error)
        raise RAGGenerationError("rag_generation_failed", sanitize_rag_error_message(error), error) from error

    rag_logger.info("modify_training_rag.retrieval_done retrieved_exercises=%s", len(retrieved_items))
    if not retrieved_items:
        message = "No retrieved exercises found for RAG training modification"
        rag_logger.warning("modify_training_rag.retrieval_empty: %s", message)
        raise RAGGenerationError("no_retrieved_exercises", message)

    exercise_context = build_exercise_context(results)
    if not exercise_context.strip():
        message = "No retrieved exercises found for RAG training modification"
        rag_logger.warning("modify_training_rag.context_empty: %s", message)
        raise RAGGenerationError("no_retrieved_exercises", message)

    user_payload = build_modify_training_user_payload(request, user_feedback, retrieval_query)
    messages = build_modify_training_plan_prompt(user_payload, exercise_context)
    rag_logger.info(
        "modify_training_rag.prompt_built message_count=%s context_chars=%s",
        len(messages),
        len(exercise_context),
    )

    try:
        rag_logger.info("modify_training_rag.llm_call_start provider=%s", settings.LLM_PROVIDER)
        llm_text = call_llm_chat(
            messages,
            response_schema=modified_training_plan_response_schema(),
        )
        rag_logger.info(
            "modify_training_rag.llm_call_done provider=%s response_chars=%s",
            settings.LLM_PROVIDER,
            len(llm_text or ""),
        )
    except Exception as error:
        log_rag_exception("modify_training_llm_call", error)
        category = classify_llm_error(error)
        raise RAGGenerationError(category, sanitize_rag_error_message(error), error) from error

    try:
        rag_response = parse_modified_training_plan_response(llm_text)
        rag_logger.info("modify_training_rag.parse_done")
    except ValidationError as error:
        log_rag_exception("modify_training_parse_validation", error)
        raise RAGGenerationError(
            "rag_schema_validation_failed",
            sanitize_rag_error_message(error),
            error,
        ) from error
    except ValueError as error:
        log_rag_exception("modify_training_parse_json", error)
        raise RAGGenerationError("invalid_llm_json", sanitize_rag_error_message(error), error) from error
    except Exception as error:
        log_rag_exception("modify_training_parse_unknown", error)
        raise RAGGenerationError("invalid_llm_json", sanitize_rag_error_message(error), error) from error

    try:
        response = modified_training_rag_response_to_legacy_response(
            rag_response,
            request,
            user_feedback,
            retrieved_items=retrieved_items,
        )
        rag_logger.info("modify_training_rag.legacy_conversion_done")
        return response
    except Exception as error:
        log_rag_exception("modify_training_legacy_conversion", error)
        raise RAGGenerationError(
            "rag_legacy_conversion_failed",
            sanitize_rag_error_message(error),
            error,
        ) from error

def fetch_retrieved_items_for_generate_training_guard(request: GenerateTrainingRequest):
    try:
        retrieval_query = build_training_rag_search_query(request)
        results = search_exercises(retrieval_query, n_results=30)
        return unpack_chroma_results(results)
    except Exception as error:
        log_rag_exception("generate_training_guard_retrieval", error)
        return []

def apply_shoulder_safety_guard_to_generated_training_result(
    result,
    request: GenerateTrainingRequest,
):
    if not isinstance(result, dict):
        return result
    plan_data = result.get("plan_data")
    if not isinstance(plan_data, dict):
        return result
    schedule = plan_data.get("schedule")
    if not isinstance(schedule, list) or not schedule:
        return result

    retrieved_items = fetch_retrieved_items_for_generate_training_guard(request)
    guard_changes, guard_warnings, _, guard_sources = (
        apply_shoulder_safety_guard_to_modified_schedule(
            schedule,
            request,
            {},
            retrieved_items,
        )
    )
    if not guard_changes and not guard_warnings:
        return result

    result["injury_warnings"] = unique_strings(
        guard_warnings,
        result.get("injury_warnings"),
    )
    result["sources"] = merge_sources_for_used_training_exercises(
        result.get("sources"),
        guard_sources,
        schedule,
    )
    plan_data["schedule"] = schedule
    result["plan_data"] = plan_data
    return result

@app.post("/generate-training-plan")
async def generate_training_plan(request: GenerateTrainingRequest):
    try:
        return generate_training_plan_rag(request)
    except Exception as rag_error:
        fallback = generate_training_plan_rule_based(request)
        fallback = apply_shoulder_safety_guard_to_generated_training_result(fallback, request)
        fallback["generation_mode"] = "rule_based_fallback"
        fallback["fallback_reason"] = safe_training_fallback_reason(rag_error)
        if settings.DEBUG:
            fallback["rag_debug_error_type"] = rag_debug_error_type(rag_error)
            fallback["rag_debug_error_message"] = sanitize_rag_error_message(rag_error)
        fallback.setdefault("status", "success")
        return fallback

@app.post("/modify-training-plan")
async def modify_training_plan(request: SmartModifyTrainingRequest):
    try:
        return generate_modified_training_plan_rag(request)
    except Exception as rag_error:
        try:
            user_feedback = resolve_training_modification_feedback(request)
            result = analyze_plan_and_suggest_modifications(
                current_plan=request.current_plan,
                user_feedback=user_feedback,
                search_func=search_exercises
            )
            result = apply_shoulder_safety_guard_to_legacy_modify_result(
                result,
                request,
                user_feedback,
            )
            result["generation_mode"] = "fallback"
            result["fallback_reason"] = safe_training_fallback_reason(rag_error)
            result.setdefault("status", "success")
            if settings.DEBUG:
                result["rag_debug_error_type"] = rag_debug_error_type(rag_error)
                result["rag_debug_error_message"] = sanitize_rag_error_message(rag_error)
            return result
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))


def normalize_term(value: str) -> str:
    if not value:
        return ""
    value = str(value).strip().lower()
    value = re.sub(r"[_\-]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value


FOOD_CATEGORY_MAP = {
    "seafood": ["fish", "tuna", "salmon", "shrimp", "cod", "sardine", "mackerel", "seafood"],
    "fish": ["fish", "tuna", "salmon", "shrimp", "cod", "sardine", "mackerel"],
    "nuts": ["peanut", "peanut butter", "almond", "walnut", "cashew", "pistachio", "hazelnut", "nuts"],
    "dairy": ["milk", "yogurt", "greek yogurt", "cheese", "labneh", "cream", "dairy"],
    "red meat": ["beef", "lean beef", "lamb", "red meat"],
    "bread": ["bread", "toast", "bun"],
    "eggs": ["egg", "eggs", "boiled egg"],
    "chicken": ["chicken", "grilled chicken breast"],
    "turkey": ["turkey", "turkey breast"],
}


def expand_blocked_terms(raw_terms):
    expanded = set()

    for term in raw_terms:
        term = normalize_term(term)
        if not term:
            continue

        expanded.add(term)

        if term.endswith("s") and len(term) > 3:
            expanded.add(term[:-1])
        else:
            expanded.add(term + "s")

        if term in FOOD_CATEGORY_MAP:
            for alias in FOOD_CATEGORY_MAP[term]:
                expanded.add(normalize_term(alias))

    return expanded


def collect_blocked_terms(preferences: dict):
    raw_terms = []

    for item in preferences.get("disliked_foods", []) or []:
        raw_terms.append(str(item))

    for key in ["food_allergies", "medical_conditions", "preferences"]:
        text = str(preferences.get(key) or "")
        if text:
            parts = re.split(r"[,\n/]+", text)
            raw_terms.extend(parts)

    return expand_blocked_terms(raw_terms)


def food_matches_restrictions(food_name: str, blocked_terms):
    name = normalize_term(food_name)
    if not name:
        return False

    if name in blocked_terms:
        return True

    for term in blocked_terms:
        if not term:
            continue
        if term in name or name in term:
            return True

    return False


def generate_nutrition_plan_rule_based(request: GenerateNutritionRequest):
    try:
        goal = normalize_term(request.user_summary.goal or "").replace(" ", "_")
        preferences = request.preferences or {}

        blocked_terms = collect_blocked_terms(preferences)
        liked_foods = [
            normalize_term(x)
            for x in (preferences.get("liked_foods", []) or [])
            if normalize_term(x)
        ]

        if goal == "higher_protein":
            blocked_terms.update(
                expand_blocked_terms([
                    "nuts",
                    "almonds",
                    "walnuts",
                    "peanut",
                    "peanut butter",
                ])
            )

        queries = []

        if goal == "muscle_gain":
            queries.extend([
                "high protein lean food",
                "healthy muscle gain food",
                "lean protein food"
            ])
        elif goal in ["fat_loss", "lower_calories", "weight_loss"]:
            queries.extend([
                "low calorie high protein food",
                "healthy low calorie food",
                "lean protein food"
            ])
        elif goal == "higher_protein":
            queries.extend([
                "high protein lean food",
                "lean protein food",
                "healthy protein rich food"
            ])
        else:
            queries.extend([
                f"{goal} healthy food" if goal else "healthy balanced food",
                "healthy balanced food"
            ])

        for liked in liked_foods[:5]:
            queries.append(liked)

        suggested_foods = []
        seen_ids = set()
        seen_names = set()

        for query in queries[:12]:
            results = search_nutrition(query, n_results=20)
            docs = results.get("documents", [[]])
            metas = results.get("metadatas", [[]])

            if not docs or not metas:
                continue

            for _, meta in zip(docs[0], metas[0]):
                food_id = meta.get("id")
                food_name = str(meta.get("name", "")).strip()
                food_name_l = normalize_term(food_name)

                if not food_id or not food_name:
                    continue

                if food_id in seen_ids or food_name_l in seen_names:
                    continue

                if food_matches_restrictions(food_name, blocked_terms):
                    continue

                calories = float(meta.get("calories", 0) or 0)
                fat = float(meta.get("fat", 0) or 0)
                protein = float(meta.get("protein", 0) or 0)
                carbs = float(meta.get("carbs", 0) or 0)

                if goal in ["fat_loss", "lower_calories", "weight_loss"]:
                    if calories > 220:
                        continue
                    if fat > 10:
                        continue

                if goal == "muscle_gain":
                    if protein < 8:
                        continue

                if goal == "higher_protein":
                    if protein < 12:
                        continue
                    if fat > 12:
                        continue
                    if calories > 250:
                        continue

                suggested_foods.append({
                    "food_id": food_id,
                    "name": food_name,
                    "calories": calories,
                    "protein": protein,
                    "carbs": carbs,
                    "fat": fat,
                    "quantity": 1
                })

                seen_ids.add(food_id)
                seen_names.add(food_name_l)

        meals = {"breakfast": [], "lunch": [], "dinner": [], "snacks": []}
        categories = ["breakfast", "lunch", "dinner", "snacks"]
        cat_idx = 0

        for food in suggested_foods[:16]:
            meals[categories[cat_idx % 4]].append(food)
            cat_idx += 1

        # final cleanup
        for meal_name in meals:
            cleaned_items = []
            for item in meals[meal_name]:
                item_name = item.get("name", "")
                item_calories = float(item.get("calories", 0) or 0)
                item_fat = float(item.get("fat", 0) or 0)
                item_protein = float(item.get("protein", 0) or 0)

                if food_matches_restrictions(item_name, blocked_terms):
                    continue

                if goal in ["fat_loss", "lower_calories", "weight_loss"]:
                    if item_calories > 220 or item_fat > 10:
                        continue

                if goal == "higher_protein":
                    if item_protein < 12:
                        continue
                    if item_fat > 12:
                        continue
                    if item_calories > 250:
                        continue

                cleaned_items.append(item)

            meals[meal_name] = cleaned_items

        total_calories = sum(item["calories"] for meal in meals.values() for item in meal)

        return {
            "plan_id": f"meal_{request.user_summary.user_id}_{int(datetime.now().timestamp())}",
            "version": 1,
            "generated_at": datetime.now().isoformat(),
            "daily_meals": [
                {"meal": "breakfast", "items": meals["breakfast"]},
                {"meal": "lunch", "items": meals["lunch"]},
                {"meal": "dinner", "items": meals["dinner"]},
                {"meal": "snacks", "items": meals["snacks"]}
            ],
            "total_daily": {
                "calories": total_calories,
                "protein": sum(item.get("protein", 0) for meal in meals.values() for item in meal),
                "carbs": sum(item.get("carbs", 0) for meal in meals.values() for item in meal),
                "fat": sum(item.get("fat", 0) for meal in meals.values() for item in meal)
            }
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def first_nutrition_preference(preferences: Dict[str, Any], keys: List[str]):
    for key in keys:
        value = preferences.get(key)
        if value not in [None, "", []]:
            return value
    return None

def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[,\n/]+", value) if item.strip()]
    return [value]

def numeric_value(value):
    parsed = safe_float(value, None)
    if parsed is not None:
        return parsed
    match = re.search(r"-?\d+(?:\.\d+)?", safe_str(value))
    if match:
        return safe_float(match.group(0), None)
    return None

def normalize_nutrition_target_macros(target_macros):
    if not target_macros:
        return None

    macros = model_to_plain_dict(target_macros)
    protein = numeric_value(first_present(macros.get("protein_g"), macros.get("protein")))
    carbs = numeric_value(first_present(macros.get("carbs_g"), macros.get("carbs")))
    fat = numeric_value(first_present(macros.get("fat_g"), macros.get("fat")))

    if protein is None and carbs is None and fat is None:
        return None

    return {
        "protein_g": protein if protein is not None else 0.0,
        "carbs_g": carbs if carbs is not None else 0.0,
        "fat_g": fat if fat is not None else 0.0,
    }

def estimate_calories_from_macros(normalized_target_macros):
    if not normalized_target_macros:
        return None
    protein = normalized_target_macros.get("protein_g")
    carbs = normalized_target_macros.get("carbs_g")
    fat = normalized_target_macros.get("fat_g")
    if protein is None or carbs is None or fat is None:
        return None
    return round((protein * 4) + (carbs * 4) + (fat * 9), 2)

def round_nutrition_totals(totals):
    return {
        "calories": round(safe_float(totals.get("calories")), 2),
        "protein": round(safe_float(totals.get("protein")), 2),
        "carbs": round(safe_float(totals.get("carbs")), 2),
        "fat": round(safe_float(totals.get("fat")), 2),
    }

def build_nutrition_rag_search_query(request: GenerateNutritionRequest):
    user = request.user_summary
    preferences = request.preferences or {}
    normalized_target_macros = normalize_nutrition_target_macros(request.target_macros)
    estimated_target_calories = estimate_calories_from_macros(normalized_target_macros)
    pieces = [
        str(user.goal or "balanced"),
        "nutrition plan foods",
    ]

    dietary_preference = first_nutrition_preference(
        preferences,
        ["dietary_preference", "diet_type", "preferred_diet", "nutrition_preference"],
    )
    if dietary_preference:
        pieces.append(str(dietary_preference))

    meal_count = first_nutrition_preference(preferences, ["meal_count", "meals_per_day", "number_of_meals"])
    if meal_count:
        pieces.append(f"{meal_count} meals per day")

    target_calories = first_nutrition_preference(
        preferences,
        ["daily_calorie_target", "target_calories", "calorie_target", "calories"],
    )
    if target_calories:
        pieces.append(f"{target_calories} calorie target")
    elif estimated_target_calories:
        pieces.append(f"{estimated_target_calories:g} calorie target")

    if normalized_target_macros:
        pieces.append(f"{normalized_target_macros['protein_g']:g}g protein")
        pieces.append(f"{normalized_target_macros['carbs_g']:g}g carbs")
        pieces.append(f"{normalized_target_macros['fat_g']:g}g fat")

    liked_foods = as_list(preferences.get("liked_foods"))
    for liked in liked_foods[:5]:
        pieces.append(str(liked))

    if preferences.get("food_allergies") or preferences.get("allergies"):
        pieces.append("allergy aware safe foods")

    return " ".join(str(piece).strip() for piece in pieces if str(piece).strip())

def build_nutrition_user_payload(request: GenerateNutritionRequest, retrieval_query: str):
    preferences = request.preferences or {}
    normalized_target_macros = normalize_nutrition_target_macros(request.target_macros)
    estimated_target_calories = estimate_calories_from_macros(normalized_target_macros)
    meal_count = first_nutrition_preference(preferences, ["meal_count", "meals_per_day", "number_of_meals"])
    allergies = as_list(first_present(preferences.get("food_allergies"), preferences.get("allergies")))
    liked_foods = as_list(preferences.get("liked_foods"))
    disliked_foods = as_list(preferences.get("disliked_foods"))

    return {
        "user_summary": model_to_plain_dict(request.user_summary),
        "preferences": preferences,
        "target_macros": model_to_plain_dict(request.target_macros) if request.target_macros else None,
        "normalized_target_macros": normalized_target_macros,
        "estimated_target_calories": estimated_target_calories,
        "target_calories": first_nutrition_preference(
            preferences,
            ["daily_calorie_target", "target_calories", "calorie_target", "calories"],
        ),
        "meal_count": meal_count,
        "dietary_preference": first_nutrition_preference(
            preferences,
            ["dietary_preference", "diet_type", "preferred_diet", "nutrition_preference"],
        ),
        "allergies": allergies,
        "liked_foods": liked_foods,
        "disliked_foods": disliked_foods,
        "retrieval_query": retrieval_query,
    }

def log_nutrition_rag_exception(stage: str, error: Exception):
    rag_logger.warning(
        "nutrition_rag.%s failed: %s: %s",
        stage,
        type(error).__name__,
        sanitize_rag_error_message(error),
    )

def safe_nutrition_fallback_reason(error: Exception):
    if isinstance(error, RAGGenerationError):
        return error.category

    message = str(error or "")
    if "OPENROUTER_API_KEY" in message or "OPENROUTER_MODEL" in message:
        return "openrouter_not_configured"
    if "GEMINI_API_KEY" in message or "GEMINI_GENERATION_MODEL" in message:
        return "gemini_generation_failed"
    if "timed out" in message.lower():
        return "gemini_generation_failed" if "Gemini" in message else "openrouter_timeout"
    if "OpenRouter request failed with status" in message:
        return "openrouter_http_error"
    if "Gemini request failed" in message or "Gemini returned" in message:
        if "empty assistant text" in message or "did not contain assistant text" in message:
            return "llm_empty_response"
        return "gemini_generation_failed"
    if "Unsupported LLM_PROVIDER" in message:
        return "llm_generation_failed"
    if "empty assistant message content" in message or "did not contain assistant message content" in message:
        return "llm_empty_response"
    if "No retrieved foods" in message:
        return "no_retrieved_foods"
    if "nutrition totals too low" in message.lower() or "below 60%" in message:
        return "nutrition_totals_too_low"
    if "No valid JSON object" in message or "No response text" in message or "JSON" in message:
        return "invalid_llm_json"
    if "validation" in message.lower():
        return "rag_schema_validation_failed"
    return "rag_generation_failed"

def food_item_prompt_score(item):
    score = distance_to_score(item.get("distance"))
    if score is not None:
        return score
    distance = safe_float(item.get("distance"), None)
    if distance is not None:
        return -distance
    return 0.0

def deduplicate_food_results_for_prompt(results):
    deduped = {}
    for item in unpack_chroma_results(results):
        meta = item.get("metadata") or {}
        name_key = normalize_term(meta.get("name") or item.get("id") or "")
        if not name_key:
            name_key = safe_str(item.get("id"))
        if not name_key:
            continue

        existing = deduped.get(name_key)
        if existing is None or food_item_prompt_score(item) > food_item_prompt_score(existing):
            deduped[name_key] = item

    return pack_chroma_results(list(deduped.values()))

def nutrition_rag_response_to_legacy_response(rag_response, request: GenerateNutritionRequest, retrieved_items=None):
    daily_meals = []
    total_daily = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
    normalized_target_macros = normalize_nutrition_target_macros(request.target_macros)
    estimated_target_calories = estimate_calories_from_macros(normalized_target_macros)

    for meal in rag_response.meals:
        items = []
        for food in meal.foods:
            quantity = food.quantity
            quantity_number = safe_float(quantity, 1.0)
            if quantity_number <= 0:
                quantity_number = 1.0

            item = {
                "food_id": food.food_id,
                "name": food.name,
                "serving_size": food.serving_size,
                "quantity": quantity,
                "calories": food.calories,
                "protein": food.protein,
                "carbs": food.carbs,
                "fat": food.fat,
                "reason": food.reason,
                "source_id": food.source_id,
            }
            items.append(item)

            total_daily["calories"] += food.calories * quantity_number
            total_daily["protein"] += food.protein * quantity_number
            total_daily["carbs"] += food.carbs * quantity_number
            total_daily["fat"] += food.fat * quantity_number

        daily_meals.append({
            "meal": meal.meal,
            "items": items,
        })

    if not daily_meals:
        raise ValueError("LLM response did not include nutrition meals")
    if not any(meal["items"] for meal in daily_meals):
        raise ValueError("LLM response did not include usable foods")

    total_daily = round_nutrition_totals(total_daily)
    rag_logger.info(
        "nutrition_rag.total_daily_recalculated calories=%s protein=%s carbs=%s fat=%s estimated_target_calories=%s",
        total_daily["calories"],
        total_daily["protein"],
        total_daily["carbs"],
        total_daily["fat"],
        estimated_target_calories,
    )

    if estimated_target_calories and total_daily["calories"] < (estimated_target_calories * 0.6):
        rag_logger.warning(
            "nutrition_rag.nutrition_totals_low_before_guard total_calories=%s threshold=%s estimated_target_calories=%s",
            total_daily["calories"],
            round(estimated_target_calories * 0.6, 2),
            estimated_target_calories,
        )

    macro_targets = normalized_target_macros or model_to_plain_dict(rag_response.macro_targets)

    response = {
        "status": rag_response.status,
        "plan_id": f"meal_{request.user_summary.user_id}_{int(datetime.now().timestamp())}",
        "version": 1,
        "generated_at": datetime.now().isoformat(),
        "daily_meals": daily_meals,
        "total_daily": total_daily,
        "daily_calorie_target": estimated_target_calories or rag_response.daily_calorie_target,
        "macro_targets": macro_targets,
        "generation_mode": "rag",
        "rag_summary": getattr(
            rag_response,
            "summary",
            "Nutrition plan generated from retrieved food context.",
        ),
        "sources": [model_to_plain_dict(source) for source in rag_response.sources],
        "allergy_warnings": rag_response.allergy_warnings,
    }
    return apply_nutrition_safety_guard_to_plan_result(
        response,
        nutrition_generation_guard_feedback(request),
        retrieved_items or [],
    )

def generate_nutrition_plan_rag(request: GenerateNutritionRequest):
    retrieval_query = build_nutrition_rag_search_query(request)
    normalized_target_macros = normalize_nutrition_target_macros(request.target_macros)
    estimated_target_calories = estimate_calories_from_macros(normalized_target_macros)
    rag_logger.info(
        "nutrition_rag.target_macros received=%s normalized=%s estimated_target_calories=%s",
        bool(request.target_macros),
        normalized_target_macros,
        estimated_target_calories,
    )
    rag_logger.info("nutrition_rag.retrieval_start user_id=%s", request.user_summary.user_id)

    try:
        results = search_nutrition(retrieval_query, n_results=30)
        retrieved_items = unpack_chroma_results(results)
    except Exception as error:
        log_nutrition_rag_exception("retrieval", error)
        raise RAGGenerationError("rag_generation_failed", sanitize_rag_error_message(error), error) from error

    rag_logger.info("nutrition_rag.retrieval_done retrieved_foods=%s", len(retrieved_items))
    if not retrieved_items:
        message = "No retrieved foods found for RAG nutrition generation"
        rag_logger.warning("nutrition_rag.retrieval_empty: %s", message)
        raise RAGGenerationError("no_retrieved_foods", message)

    prompt_results = deduplicate_food_results_for_prompt(results)
    deduped_items = unpack_chroma_results(prompt_results)
    rag_logger.info(
        "nutrition_rag.prompt_dedup_done original_foods=%s deduped_foods=%s",
        len(retrieved_items),
        len(deduped_items),
    )
    if not deduped_items:
        message = "No retrieved foods found for RAG nutrition generation"
        rag_logger.warning("nutrition_rag.dedup_empty: %s", message)
        raise RAGGenerationError("no_retrieved_foods", message)

    food_context = build_food_context(prompt_results)
    if not food_context.strip():
        message = "No retrieved foods found for RAG nutrition generation"
        rag_logger.warning("nutrition_rag.context_empty: %s", message)
        raise RAGGenerationError("no_retrieved_foods", message)

    user_payload = build_nutrition_user_payload(request, retrieval_query)
    messages = build_nutrition_plan_prompt(user_payload, food_context)
    rag_logger.info(
        "nutrition_rag.prompt_built message_count=%s context_chars=%s",
        len(messages),
        len(food_context),
    )

    try:
        rag_logger.info("nutrition_rag.llm_call_start provider=%s", settings.LLM_PROVIDER)
        llm_text = call_llm_chat(messages, response_schema=nutrition_plan_response_schema())
        rag_logger.info("nutrition_rag.llm_call_done provider=%s response_chars=%s", settings.LLM_PROVIDER, len(llm_text or ""))
    except Exception as error:
        log_nutrition_rag_exception("llm_call", error)
        category = classify_llm_error(error)
        raise RAGGenerationError(category, sanitize_rag_error_message(error), error) from error

    try:
        rag_response = parse_nutrition_plan_response(llm_text)
        rag_logger.info("nutrition_rag.parse_done")
    except ValidationError as error:
        log_nutrition_rag_exception("parse_validation", error)
        raise RAGGenerationError(
            "rag_schema_validation_failed",
            sanitize_rag_error_message(error),
            error,
        ) from error
    except ValueError as error:
        log_nutrition_rag_exception("parse_json", error)
        raise RAGGenerationError("invalid_llm_json", sanitize_rag_error_message(error), error) from error
    except Exception as error:
        log_nutrition_rag_exception("parse_unknown", error)
        raise RAGGenerationError("invalid_llm_json", sanitize_rag_error_message(error), error) from error

    try:
        response = nutrition_rag_response_to_legacy_response(
            rag_response,
            request,
            retrieved_items=deduped_items,
        )
        rag_logger.info("nutrition_rag.legacy_conversion_done")
        return response
    except RAGGenerationError:
        raise
    except Exception as error:
        log_nutrition_rag_exception("legacy_conversion", error)
        raise RAGGenerationError(
            "rag_legacy_conversion_failed",
            sanitize_rag_error_message(error),
            error,
        ) from error

@app.post("/generate-nutrition-plan")
async def generate_nutrition_plan(request: GenerateNutritionRequest):
    try:
        return generate_nutrition_plan_rag(request)
    except Exception as rag_error:
        fallback = generate_nutrition_plan_rule_based(request)
        fallback = apply_nutrition_safety_guard_to_plan_result(
            fallback,
            nutrition_generation_guard_feedback(request),
            fetch_retrieved_items_for_generate_nutrition_guard(request),
        )
        fallback["generation_mode"] = "rule_based_fallback"
        fallback["fallback_reason"] = safe_nutrition_fallback_reason(rag_error)
        if settings.DEBUG:
            fallback["rag_debug_error_type"] = rag_debug_error_type(rag_error)
            fallback["rag_debug_error_message"] = sanitize_rag_error_message(rag_error)
        fallback.setdefault("status", "success")
        return fallback


def nutrition_plan_id_from_request(request: SmartModifyNutritionRequest):
    return request.current_plan_id or request.current_plan.get("plan_id", "unknown")

def next_nutrition_plan_version(current_plan: Dict[str, Any]):
    try:
        return int(current_plan.get("version", 1)) + 1
    except (TypeError, ValueError):
        return 2

def resolve_nutrition_modification_feedback(request: SmartModifyNutritionRequest):
    feedback = model_to_plain_dict(request.user_feedback) if request.user_feedback else {}
    preferences = model_to_plain_dict(request.preferences) if request.preferences else {}

    for key, value in preferences.items():
        if key not in feedback or feedback.get(key) in [None, "", []]:
            feedback[key] = value

    root_modification = request.modification_request
    if isinstance(root_modification, dict):
        for key, value in root_modification.items():
            if key not in feedback or feedback.get(key) in [None, "", []]:
                feedback[key] = value

    root_modification_text = normalize_training_modification_text(root_modification)
    if root_modification_text and not feedback.get("modification_request"):
        feedback["modification_request"] = root_modification_text

    if not feedback.get("modification_request"):
        inline_text = normalize_training_modification_text({
            "reason": feedback.get("reason"),
            "notes": feedback.get("notes"),
            "request": feedback.get("request"),
            "message": feedback.get("message"),
        })
        if inline_text:
            feedback["modification_request"] = inline_text

    if request.target_macros and not feedback.get("target_macros"):
        feedback["target_macros"] = model_to_plain_dict(request.target_macros)

    if request.user_summary.goal and not feedback.get("goal"):
        feedback["goal"] = request.user_summary.goal

    if "allergies" in feedback and not feedback.get("food_allergies"):
        feedback["food_allergies"] = feedback.get("allergies")
    if "food_allergies" in feedback and not feedback.get("allergies"):
        feedback["allergies"] = feedback.get("food_allergies")

    for key in ["liked_foods", "disliked_foods"]:
        if key in feedback:
            feedback[key] = as_list(feedback.get(key))

    return feedback

def modified_nutrition_target_macros(request: SmartModifyNutritionRequest, user_feedback: Dict[str, Any]):
    return normalize_nutrition_target_macros(
        request.target_macros or user_feedback.get("target_macros")
    )

def extract_nutrition_daily_meals(current_plan: Dict[str, Any]):
    if not isinstance(current_plan, dict):
        return []
    if isinstance(current_plan.get("daily_meals"), list):
        return current_plan.get("daily_meals") or []
    if isinstance(current_plan.get("meals"), list):
        return current_plan.get("meals") or []
    plan_data = current_plan.get("plan_data")
    if isinstance(plan_data, dict):
        if isinstance(plan_data.get("daily_meals"), list):
            return plan_data.get("daily_meals") or []
        if isinstance(plan_data.get("meals"), list):
            return plan_data.get("meals") or []
    return []

def build_modify_nutrition_rag_search_query(
    request: SmartModifyNutritionRequest,
    user_feedback: Dict[str, Any],
):
    user = request.user_summary
    normalized_target_macros = modified_nutrition_target_macros(request, user_feedback)
    estimated_target_calories = estimate_calories_from_macros(normalized_target_macros)
    pieces = [
        str(user.goal or user_feedback.get("goal") or "balanced"),
        "nutrition plan modification foods",
        normalize_training_modification_text(user_feedback.get("modification_request")),
    ]

    meal_count = first_nutrition_preference(
        user_feedback,
        ["meal_count", "meals_per_day", "number_of_meals"],
    )
    if meal_count:
        pieces.append(f"{meal_count} meals per day")

    if estimated_target_calories:
        pieces.append(f"{estimated_target_calories:g} calorie target")
    if normalized_target_macros:
        pieces.append(f"{normalized_target_macros['protein_g']:g}g protein")
        pieces.append(f"{normalized_target_macros['carbs_g']:g}g carbs")
        pieces.append(f"{normalized_target_macros['fat_g']:g}g fat")

    allergies = as_list(first_present(user_feedback.get("food_allergies"), user_feedback.get("allergies")))
    for allergy in allergies[:5]:
        pieces.append(f"{allergy} allergy safe foods")

    for liked in as_list(user_feedback.get("liked_foods"))[:5]:
        pieces.append(str(liked))
    for disliked in as_list(user_feedback.get("disliked_foods"))[:5]:
        pieces.append(f"avoid {disliked}")

    for meal in extract_nutrition_daily_meals(request.current_plan)[:6]:
        if not isinstance(meal, dict):
            continue
        if meal.get("meal") or meal.get("meal_type"):
            pieces.append(str(meal.get("meal") or meal.get("meal_type")))
        for item in (meal.get("items") or meal.get("foods") or [])[:4]:
            if isinstance(item, dict) and item.get("name"):
                pieces.append(str(item.get("name")))

    return " ".join(str(piece).strip() for piece in pieces if str(piece).strip())

def build_modify_nutrition_user_payload(
    request: SmartModifyNutritionRequest,
    user_feedback: Dict[str, Any],
    retrieval_query: str,
):
    normalized_target_macros = modified_nutrition_target_macros(request, user_feedback)
    estimated_target_calories = estimate_calories_from_macros(normalized_target_macros)
    allergies = as_list(first_present(user_feedback.get("food_allergies"), user_feedback.get("allergies")))
    liked_foods = as_list(user_feedback.get("liked_foods"))
    disliked_foods = as_list(user_feedback.get("disliked_foods"))

    return {
        "current_plan_id": nutrition_plan_id_from_request(request),
        "user_summary": model_to_plain_dict(request.user_summary),
        "preferences": model_to_plain_dict(request.preferences) if request.preferences else {},
        "user_feedback": user_feedback,
        "modification_request": normalize_training_modification_text(
            user_feedback.get("modification_request")
        ),
        "target_macros": model_to_plain_dict(request.target_macros) if request.target_macros else user_feedback.get("target_macros"),
        "normalized_target_macros": normalized_target_macros,
        "estimated_target_calories": estimated_target_calories,
        "meal_count": first_nutrition_preference(
            user_feedback,
            ["meal_count", "meals_per_day", "number_of_meals"],
        ),
        "allergies": allergies,
        "liked_foods": liked_foods,
        "disliked_foods": disliked_foods,
        "current_plan": request.current_plan,
        "current_daily_meals": extract_nutrition_daily_meals(request.current_plan),
        "retrieval_query": retrieval_query,
    }

def nutrition_item_quantity_number(item):
    quantity = item.get("quantity", 1) if isinstance(item, dict) else 1
    value = numeric_value(quantity)
    if value is None or value <= 0:
        return 1.0
    return value

def recalculate_nutrition_daily_totals(daily_meals):
    total_daily = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
    for meal in daily_meals or []:
        for item in meal.get("items", []) or []:
            quantity = nutrition_item_quantity_number(item)
            total_daily["calories"] += safe_float(item.get("calories")) * quantity
            total_daily["protein"] += safe_float(item.get("protein")) * quantity
            total_daily["carbs"] += safe_float(item.get("carbs")) * quantity
            total_daily["fat"] += safe_float(item.get("fat")) * quantity
    return round_nutrition_totals(total_daily)

def nutrition_source_id_from_item(item):
    if not isinstance(item, dict):
        return None
    return item.get("source_id") or item.get("food_id") or item.get("nutrition_id")

def used_nutrition_source_ids(daily_meals):
    return {
        str(source_id)
        for meal in daily_meals or []
        for item in meal.get("items", []) or []
        for source_id in [nutrition_source_id_from_item(item)]
        if source_id is not None
    }

def retrieved_food_source_id(item):
    meta = item.get("metadata") or {}
    return meta.get("id") if meta.get("id") not in [None, ""] else item.get("id")

def retrieved_food_source_map(retrieved_items):
    source_map = {}
    for item in retrieved_items or []:
        meta = item.get("metadata") or {}
        source_id = retrieved_food_source_id(item)
        if source_id in [None, ""]:
            continue
        source_map[str(source_id)] = {
            "source_id": source_id,
            "source_table": "foods",
            "source_name": safe_str(meta.get("name"), "Food"),
            "score": distance_to_score(item.get("distance")),
            "reason_used": "Used in modified nutrition plan from retrieved food context.",
        }
    return source_map

def merge_sources_for_used_nutrition_foods(existing_sources, guard_sources, daily_meals, retrieved_items=None):
    used_source_ids = used_nutrition_source_ids(daily_meals)
    source_lookup = retrieved_food_source_map(retrieved_items or [])
    merged = []
    seen = set()

    for source in (existing_sources or []) + (guard_sources or []):
        source_dict = model_to_plain_dict(source)
        source_id = str(source_dict.get("source_id"))
        if source_id not in used_source_ids or source_id in seen:
            continue
        merged.append(source_dict)
        seen.add(source_id)

    for source_id in sorted(used_source_ids):
        if source_id in seen or source_id not in source_lookup:
            continue
        merged.append(source_lookup[source_id])
        seen.add(source_id)

    return merged

def nutrition_food_to_legacy_item(food):
    return {
        "food_id": food.food_id,
        "name": food.name,
        "serving_size": food.serving_size,
        "quantity": food.quantity,
        "calories": food.calories,
        "protein": food.protein,
        "carbs": food.carbs,
        "fat": food.fat,
        "reason": food.reason,
        "source_id": food.source_id,
    }

def retrieved_food_to_candidate(item):
    meta = item.get("metadata") or {}
    source_id = retrieved_food_source_id(item)
    name = safe_str(meta.get("name"), "").strip()
    if source_id in [None, ""] or not name:
        return None
    return {
        "food_id": source_id,
        "source_id": source_id,
        "name": name,
        "serving_size": meta.get("serving_size"),
        "quantity": 1,
        "calories": safe_float(meta.get("calories")),
        "protein": safe_float(meta.get("protein")),
        "carbs": safe_float(meta.get("carbs")),
        "fat": safe_float(meta.get("fat")),
        "score": distance_to_score(item.get("distance")),
    }

def nutrition_candidate_rank(candidate, user_feedback):
    name = normalize_term(candidate.get("name"))
    liked_foods = [normalize_term(food) for food in as_list(user_feedback.get("liked_foods"))]
    goal = normalize_term(user_feedback.get("goal", "")).replace(" ", "_")
    calories = safe_float(candidate.get("calories"))
    protein = safe_float(candidate.get("protein"))
    fat = safe_float(candidate.get("fat"))
    retrieved_score = safe_float(candidate.get("score"), 0.0) or 0.0

    score = retrieved_score
    if any(liked and liked in name for liked in liked_foods):
        score += 5
    score += min(protein / 10, 4)
    if goal in {"fat_loss", "weight_loss", "lower_calories"}:
        if calories <= 250:
            score += 2
        if fat <= 10:
            score += 1
    return (-score, calories, safe_str(candidate.get("name")).lower())

def is_nutrition_nut_candidate(candidate):
    name = normalize_term(candidate.get("name"))
    return any(
        marker in name
        for marker in ["almond", "walnut", "cashew", "pistachio", "hazelnut", "peanut", "nut butter"]
    )

def has_nutrition_nut_item(used_names):
    return any(
        any(marker in normalize_term(name) for marker in ["almond", "walnut", "cashew", "pistachio", "hazelnut"])
        for name in used_names or []
    )

def requested_nutrition_meal_count(user_feedback):
    value = first_nutrition_preference(
        user_feedback,
        ["meal_count", "meals_per_day", "number_of_meals"],
    )
    parsed = numeric_value(value)
    if parsed is None:
        return None
    parsed = int(parsed)
    if parsed <= 0:
        return None
    return min(parsed, 8)

def ensure_nutrition_meal_count(daily_meals, user_feedback):
    requested_count = requested_nutrition_meal_count(user_feedback)
    if not requested_count or len(daily_meals) >= requested_count:
        return []

    default_names = ["Breakfast", "Lunch", "Dinner", "Snack"]
    existing_names = {normalize_term(meal.get("meal")) for meal in daily_meals}
    added_names = []

    while len(daily_meals) < requested_count:
        if len(daily_meals) < len(default_names):
            candidate_name = default_names[len(daily_meals)]
        else:
            candidate_name = f"Meal {len(daily_meals) + 1}"

        if normalize_term(candidate_name) in existing_names:
            candidate_name = f"Meal {len(daily_meals) + 1}"

        daily_meals.append({"meal": candidate_name, "items": []})
        existing_names.add(normalize_term(candidate_name))
        added_names.append(candidate_name)

    return added_names

def select_nutrition_meal_for_addition(daily_meals):
    if not daily_meals:
        return None
    return min(
        daily_meals,
        key=lambda meal: (len(meal.get("items", []) or []), safe_str(meal.get("meal")).lower()),
    )

def nutrition_macro_fill_rank(
    candidate,
    user_feedback,
    total_daily=None,
    target_macros=None,
    target_calories=None,
    used_names=None,
):
    if not total_daily or not target_macros:
        return nutrition_candidate_rank(candidate, user_feedback)

    current = round_nutrition_totals(total_daily)
    calories = safe_float(candidate.get("calories"))
    protein = safe_float(candidate.get("protein"))
    carbs = safe_float(candidate.get("carbs"))
    fat = safe_float(candidate.get("fat"))
    name = normalize_term(candidate.get("name"))
    liked_foods = [normalize_term(food) for food in as_list(user_feedback.get("liked_foods"))]
    retrieved_score = safe_float(candidate.get("score"), 0.0) or 0.0

    target_protein = safe_float(target_macros.get("protein_g"))
    target_carbs = safe_float(target_macros.get("carbs_g"))
    target_fat = safe_float(target_macros.get("fat_g"))
    target_calories = safe_float(target_calories)
    used_names = used_names or set()
    is_nut = is_nutrition_nut_candidate(candidate)

    before_deficits = {
        "calories": max(target_calories - current["calories"], 0.0),
        "protein": max(target_protein - current["protein"], 0.0),
        "carbs": max(target_carbs - current["carbs"], 0.0),
        "fat": max(target_fat - current["fat"], 0.0),
    }
    after_deficits = {
        "calories": max(target_calories - (current["calories"] + calories), 0.0),
        "protein": max(target_protein - (current["protein"] + protein), 0.0),
        "carbs": max(target_carbs - (current["carbs"] + carbs), 0.0),
        "fat": max(target_fat - (current["fat"] + fat), 0.0),
    }

    improvement = 0.0
    improvement += (before_deficits["calories"] - after_deficits["calories"]) / 80
    improvement += (before_deficits["protein"] - after_deficits["protein"]) / 8
    improvement += (before_deficits["carbs"] - after_deficits["carbs"]) / 8
    improvement += (before_deficits["fat"] - after_deficits["fat"]) / 3

    if any(liked and liked in name for liked in liked_foods):
        improvement += 5
    if before_deficits["carbs"] > 30 and carbs >= 15:
        improvement += 2
    if before_deficits["fat"] > 8 and fat >= 5:
        improvement += 1.5
    if before_deficits["protein"] <= 10 and protein > 35:
        improvement -= 1

    overage_penalty = 0.0
    overage_penalty += max((current["protein"] + protein) - (target_protein * 1.25), 0.0) / 10
    overage_penalty += max((current["fat"] + fat) - target_fat, 0.0) / 2
    overage_penalty += max((current["fat"] + fat) - (target_fat * 1.25), 0.0) / 5
    overage_penalty += max((current["calories"] + calories) - (target_calories * 1.1), 0.0) / 100

    if current["fat"] >= (target_fat * 0.85):
        overage_penalty += max(fat - 6, 0.0) / 2
    elif fat > max(before_deficits["fat"] * 1.15, 12):
        overage_penalty += (fat - before_deficits["fat"]) / 3

    if is_nut:
        if current["fat"] >= (target_fat * 0.70):
            overage_penalty += 8
        if has_nutrition_nut_item(used_names):
            overage_penalty += 12
        if before_deficits["carbs"] > 20 or before_deficits["protein"] > 10:
            overage_penalty += 12
        if "butter" in name:
            overage_penalty += 4

    if carbs >= 20 and before_deficits["carbs"] > 30 and fat <= 12:
        improvement += 3
    if protein >= 20 and before_deficits["protein"] > 15 and fat <= 12:
        improvement += 2

    score = improvement + retrieved_score - overage_penalty
    return (-score, abs(after_deficits["calories"]), safe_str(candidate.get("name")).lower())

def find_safe_nutrition_replacement(
    retrieved_items,
    used_source_ids,
    used_names,
    blocked_terms,
    user_feedback,
    total_daily=None,
    target_macros=None,
    target_calories=None,
):
    ranked_candidates = []
    for item in retrieved_items or []:
        candidate = retrieved_food_to_candidate(item)
        if not candidate:
            continue
        candidate_id = str(candidate.get("source_id"))
        candidate_name = normalize_term(candidate.get("name"))
        if candidate_id in used_source_ids or candidate_name in used_names:
            continue
        if food_matches_restrictions(candidate.get("name"), blocked_terms):
            continue
        ranked_candidates.append((
            nutrition_macro_fill_rank(
                candidate,
                user_feedback,
                total_daily=total_daily,
                target_macros=target_macros,
                target_calories=target_calories,
                used_names=used_names,
            ),
            candidate,
        ))

    if not ranked_candidates:
        return None

    ranked_candidates.sort(key=lambda item: item[0])
    return ranked_candidates[0][1]

def nutrition_item_has_macro_data(item):
    if not isinstance(item, dict):
        return False
    return any(safe_float(item.get(key), 0.0) > 0 for key in ["calories", "protein", "carbs", "fat"])

def find_matching_retrieved_food(item_name, retrieved_items, blocked_terms):
    item_name_norm = normalize_term(item_name)
    if not item_name_norm:
        return None

    ranked_candidates = []
    for item in retrieved_items or []:
        candidate = retrieved_food_to_candidate(item)
        if not candidate:
            continue
        candidate_name = normalize_term(candidate.get("name"))
        if food_matches_restrictions(candidate.get("name"), blocked_terms):
            continue
        if candidate_name == item_name_norm:
            match_rank = 0
        elif item_name_norm in candidate_name or candidate_name in item_name_norm:
            match_rank = 1
        else:
            item_tokens = {token for token in item_name_norm.split() if len(token) > 3}
            candidate_tokens = {token for token in candidate_name.split() if len(token) > 3}
            if not item_tokens.intersection(candidate_tokens):
                continue
            match_rank = 2
        ranked_candidates.append((match_rank, nutrition_candidate_rank(candidate, {}), candidate))

    if not ranked_candidates:
        return None

    ranked_candidates.sort(key=lambda item: item[0:2])
    return ranked_candidates[0][2]

def enrich_existing_nutrition_item_from_retrieval(item, retrieved_items, blocked_terms):
    if not isinstance(item, dict) or item.get("source_id") not in [None, ""]:
        return item, None
    if nutrition_item_has_macro_data(item):
        return item, None

    candidate = find_matching_retrieved_food(item.get("name"), retrieved_items, blocked_terms)
    if not candidate:
        return item, None

    enriched_item = {
        "food_id": candidate["food_id"],
        "name": candidate["name"],
        "serving_size": candidate.get("serving_size"),
        "quantity": item.get("quantity", 1),
        "calories": candidate["calories"],
        "protein": candidate["protein"],
        "carbs": candidate["carbs"],
        "fat": candidate["fat"],
        "reason": "Matched existing unsourced food to retrieved nutrition data.",
        "source_id": candidate["source_id"],
    }
    return enriched_item, candidate

def nutrition_totals_need_fallback_fill(total_daily, target_macros, target_calorie_floor, target_calories):
    if not target_macros or not target_calories:
        return False
    if safe_float(total_daily.get("calories")) >= (target_calories * 1.1):
        return False
    if safe_float(total_daily.get("calories")) < target_calorie_floor:
        return True
    if safe_float(total_daily.get("protein")) < (safe_float(target_macros.get("protein_g")) * 0.85):
        return True
    if safe_float(total_daily.get("carbs")) < (safe_float(target_macros.get("carbs_g")) * 0.60):
        return True
    if safe_float(total_daily.get("fat")) < (safe_float(target_macros.get("fat_g")) * 0.60):
        return True
    return False

def subtract_nutrition_item_from_totals(total_daily, item):
    quantity = nutrition_item_quantity_number(item)
    return round_nutrition_totals({
        "calories": safe_float(total_daily.get("calories")) - (safe_float(item.get("calories")) * quantity),
        "protein": safe_float(total_daily.get("protein")) - (safe_float(item.get("protein")) * quantity),
        "carbs": safe_float(total_daily.get("carbs")) - (safe_float(item.get("carbs")) * quantity),
        "fat": safe_float(total_daily.get("fat")) - (safe_float(item.get("fat")) * quantity),
    })

def nutrition_excess_protein_removal_rank(item, occurrence_count):
    name = normalize_term(item.get("name"))
    protein = safe_float(item.get("protein"))
    carbs = safe_float(item.get("carbs"))
    fat = safe_float(item.get("fat"))
    duplicate_bonus = 30 if occurrence_count > 1 else 0
    added_bonus = 15 if "added from retrieved foods" in normalize_term(item.get("reason")) else 0
    lean_protein_bonus = 10 if protein >= 20 and carbs <= 5 and fat <= 8 else 0
    return -(
        protein
        + duplicate_bonus
        + added_bonus
        + lean_protein_bonus
        - (carbs * 0.15)
        - (fat * 0.1)
    ), name

def trim_excess_nutrition_protein(daily_meals, target_macros, target_calories):
    if not target_macros or not target_calories:
        return [], []

    target_protein = safe_float(target_macros.get("protein_g"))
    if target_protein <= 0:
        return [], []

    total_daily = recalculate_nutrition_daily_totals(daily_meals)
    protein_ceiling = target_protein * 1.30
    protein_floor = target_protein * 0.85
    calorie_floor = safe_float(target_calories) * 0.75
    if total_daily["protein"] <= protein_ceiling:
        return [], []

    changes = []
    warnings = []

    while total_daily["protein"] > protein_ceiling:
        name_counts = {}
        for meal in daily_meals:
            for item in meal.get("items", []) or []:
                name = normalize_term(item.get("name"))
                if name:
                    name_counts[name] = name_counts.get(name, 0) + 1

        candidates = []
        for meal_index, meal in enumerate(daily_meals):
            items = meal.get("items", []) or []
            if len(items) <= 1:
                continue
            for item_index, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                if nutrition_source_id_from_item(item) is None:
                    continue
                if safe_float(item.get("protein")) < 10:
                    continue

                after = subtract_nutrition_item_from_totals(total_daily, item)
                if after["calories"] < calorie_floor:
                    continue
                if after["protein"] < protein_floor:
                    continue
                rank = nutrition_excess_protein_removal_rank(
                    item,
                    name_counts.get(normalize_term(item.get("name")), 1),
                )
                candidates.append((rank, meal_index, item_index, item, after))

        if not candidates:
            warnings.append(
                f"Protein remains above target because reducing more foods would drop calories below {calorie_floor:g} or protein below {protein_floor:g}."
            )
            break

        candidates.sort(key=lambda candidate: candidate[0])
        _, meal_index, item_index, item, after = candidates[0]
        removed_name = safe_str(item.get("name"), "high-protein food")
        daily_meals[meal_index]["items"].pop(item_index)
        total_daily = after
        changes.append(
            f"Removed '{removed_name}' to keep protein closer to the requested target."
        )

    return changes, warnings

def fetch_retrieved_items_for_modify_nutrition_guard(
    request: SmartModifyNutritionRequest,
    user_feedback: Dict[str, Any],
):
    try:
        retrieval_query = build_modify_nutrition_rag_search_query(request, user_feedback)
        results = search_nutrition(retrieval_query, n_results=30)
        return unpack_chroma_results(deduplicate_food_results_for_prompt(results))
    except Exception as error:
        log_nutrition_rag_exception("modify_nutrition_guard_retrieval", error)
        return []

def nutrition_generation_guard_feedback(request: GenerateNutritionRequest):
    feedback = model_to_plain_dict(request.preferences) if request.preferences else {}
    if request.target_macros:
        feedback["target_macros"] = model_to_plain_dict(request.target_macros)
    if request.user_summary.goal and not feedback.get("goal"):
        feedback["goal"] = request.user_summary.goal
    if "allergies" in feedback and not feedback.get("food_allergies"):
        feedback["food_allergies"] = feedback.get("allergies")
    if "food_allergies" in feedback and not feedback.get("allergies"):
        feedback["allergies"] = feedback.get("food_allergies")
    for key in ["liked_foods", "disliked_foods"]:
        if key in feedback:
            feedback[key] = as_list(feedback.get(key))
    return feedback

def fetch_retrieved_items_for_generate_nutrition_guard(request: GenerateNutritionRequest):
    try:
        retrieval_query = build_nutrition_rag_search_query(request)
        results = search_nutrition(retrieval_query, n_results=30)
        return unpack_chroma_results(deduplicate_food_results_for_prompt(results))
    except Exception as error:
        log_nutrition_rag_exception("generate_nutrition_guard_retrieval", error)
        return []

def apply_nutrition_safety_guard_to_plan_result(
    result,
    user_feedback: Dict[str, Any],
    retrieved_items,
):
    if not isinstance(result, dict):
        return result

    guard_changes, guard_warnings, guard_recommendations, guard_sources = (
        apply_nutrition_safety_guard_to_modified_plan(
            result,
            user_feedback,
            retrieved_items,
        )
    )
    daily_meals = extract_nutrition_daily_meals(result)
    result["changes_summary"] = unique_strings(
        result.get("changes_summary"),
        guard_changes,
        limit=20,
    )
    result["recommendations"] = unique_strings(
        result.get("recommendations"),
        guard_recommendations,
        limit=8,
    )
    result["allergy_warnings"] = unique_strings(
        result.get("allergy_warnings"),
        guard_warnings,
    )
    result["sources"] = merge_sources_for_used_nutrition_foods(
        result.get("sources"),
        guard_sources,
        daily_meals,
        retrieved_items=retrieved_items,
    )
    if isinstance(result.get("total_daily"), dict):
        result["total_daily"] = result.get("total_daily")
    if isinstance(result.get("macro_targets"), dict):
        result["macro_targets"] = result.get("macro_targets")
    return result

def apply_nutrition_safety_guard_to_modified_plan(
    modified_plan,
    user_feedback: Dict[str, Any],
    retrieved_items,
):
    if not isinstance(modified_plan, dict):
        return [], [], [], []

    daily_meals = []
    for meal in extract_nutrition_daily_meals(modified_plan):
        if not isinstance(meal, dict):
            continue
        daily_meals.append({
            "meal": meal.get("meal") or meal.get("meal_type") or "Meal",
            "items": meal.get("items") or meal.get("foods") or [],
        })

    blocked_terms = collect_blocked_terms(user_feedback)
    allergies = as_list(first_present(user_feedback.get("food_allergies"), user_feedback.get("allergies")))
    guard_changes = []
    guard_warnings = []
    guard_sources = []
    added_meal_names = ensure_nutrition_meal_count(daily_meals, user_feedback)
    if added_meal_names:
        guard_changes.append(
            f"Added meal slots to respect requested meal count: {', '.join(added_meal_names)}."
        )

    used_source_ids = used_nutrition_source_ids(daily_meals)
    used_names = {
        normalize_term(item.get("name"))
        for meal in daily_meals
        for item in meal.get("items", [])
        if isinstance(item, dict) and item.get("name")
    }

    if allergies:
        guard_warnings.append(
            f"Food allergy noted: avoided foods conflicting with {', '.join(str(item) for item in allergies)}."
        )

    for meal in daily_meals:
        safe_items = []
        for item in meal.get("items", []) or []:
            if not isinstance(item, dict):
                continue
            item_name = safe_str(item.get("name"), "restricted food")
            if blocked_terms and food_matches_restrictions(item_name, blocked_terms):
                replacement = find_safe_nutrition_replacement(
                    retrieved_items,
                    used_source_ids,
                    used_names,
                    blocked_terms,
                    user_feedback,
                )
                if replacement:
                    replacement_item = {
                        "food_id": replacement["food_id"],
                        "name": replacement["name"],
                        "serving_size": replacement.get("serving_size"),
                        "quantity": 1,
                        "calories": replacement["calories"],
                        "protein": replacement["protein"],
                        "carbs": replacement["carbs"],
                        "fat": replacement["fat"],
                        "reason": f"Safety replacement for {item_name}: avoided allergy or disliked food conflict.",
                        "source_id": replacement["source_id"],
                    }
                    safe_items.append(replacement_item)
                    used_source_ids.add(str(replacement["source_id"]))
                    used_names.add(normalize_term(replacement["name"]))
                    guard_sources.append({
                        "source_id": replacement["source_id"],
                        "source_table": "foods",
                        "source_name": replacement["name"],
                        "score": replacement.get("score"),
                        "reason_used": "Used by nutrition safety guard as a safer retrieved replacement.",
                    })
                    guard_changes.append(
                        f"Replaced '{item_name}' with '{replacement['name']}' to avoid allergy or disliked food conflict."
                    )
                else:
                    guard_changes.append(
                        f"Removed '{item_name}' because it conflicts with allergy or disliked food restrictions."
                    )
                    guard_warnings.append(
                        f"Removed '{item_name}' because no safe retrieved replacement was available."
                    )
                continue

            enriched_item, matched_candidate = enrich_existing_nutrition_item_from_retrieval(
                item,
                retrieved_items,
                blocked_terms,
            )
            if matched_candidate:
                safe_items.append(enriched_item)
                used_source_ids.add(str(matched_candidate["source_id"]))
                used_names.add(normalize_term(matched_candidate["name"]))
                guard_sources.append({
                    "source_id": matched_candidate["source_id"],
                    "source_table": "foods",
                    "source_name": matched_candidate["name"],
                    "score": matched_candidate.get("score"),
                    "reason_used": "Matched existing unsourced food to retrieved nutrition data.",
                })
                guard_changes.append(
                    f"Matched existing '{item_name}' to retrieved food '{matched_candidate['name']}' for nutrition totals."
                )
            else:
                safe_items.append(item)

        meal["items"] = safe_items

    normalized_target_macros = normalize_nutrition_target_macros(user_feedback.get("target_macros"))
    estimated_target_calories = estimate_calories_from_macros(normalized_target_macros)
    target_calorie_floor = estimated_target_calories * 0.75 if estimated_target_calories else None
    if normalized_target_macros:
        modified_plan["macro_targets"] = normalized_target_macros
    if estimated_target_calories:
        modified_plan["daily_calorie_target"] = estimated_target_calories

    if daily_meals and estimated_target_calories:
        total_daily = recalculate_nutrition_daily_totals(daily_meals)
        additions = 0
        while (
            nutrition_totals_need_fallback_fill(
                total_daily,
                normalized_target_macros,
                target_calorie_floor,
                estimated_target_calories,
            )
            and additions < 12
        ):
            replacement = find_safe_nutrition_replacement(
                retrieved_items,
                used_source_ids,
                used_names,
                blocked_terms,
                user_feedback,
                total_daily=total_daily,
                target_macros=normalized_target_macros,
                target_calories=estimated_target_calories,
            )
            if not replacement:
                break

            added_item = {
                "food_id": replacement["food_id"],
                "name": replacement["name"],
                "serving_size": replacement.get("serving_size"),
                "quantity": 1,
                "calories": replacement["calories"],
                "protein": replacement["protein"],
                "carbs": replacement["carbs"],
                "fat": replacement["fat"],
                "reason": "Added from retrieved foods to keep modified nutrition totals closer to the requested target.",
                "source_id": replacement["source_id"],
            }
            target_meal = select_nutrition_meal_for_addition(daily_meals)
            if target_meal is None:
                break
            target_meal.setdefault("items", []).append(added_item)
            used_source_ids.add(str(replacement["source_id"]))
            used_names.add(normalize_term(replacement["name"]))
            guard_sources.append({
                "source_id": replacement["source_id"],
                "source_table": "foods",
                "source_name": replacement["name"],
                "score": replacement.get("score"),
                "reason_used": "Added by nutrition safety guard to keep totals closer to target.",
            })
            guard_changes.append(
                f"Added '{replacement['name']}' to keep nutrition totals closer to the requested target."
            )
            total_daily = recalculate_nutrition_daily_totals(daily_meals)
            additions += 1

        if nutrition_totals_need_fallback_fill(
            total_daily,
            normalized_target_macros,
            target_calorie_floor,
            estimated_target_calories,
        ):
            limited_message = (
                "Macro completion was limited by available safe retrieved foods; "
                f"fallback calories reached {total_daily['calories']:g} of target {estimated_target_calories:g}."
            )
            guard_changes.append(limited_message)
            guard_warnings.append(limited_message)

        trim_changes, trim_warnings = trim_excess_nutrition_protein(
            daily_meals,
            normalized_target_macros,
            estimated_target_calories,
        )
        guard_changes.extend(trim_changes)
        guard_warnings.extend(trim_warnings)

    total_daily = recalculate_nutrition_daily_totals(daily_meals)
    modified_plan["daily_meals"] = daily_meals
    modified_plan["total_daily"] = total_daily
    return guard_changes, guard_warnings, [], guard_sources

def apply_nutrition_safety_guard_to_legacy_modify_result(
    result,
    request: SmartModifyNutritionRequest,
    user_feedback: Dict[str, Any],
):
    if not isinstance(result, dict):
        return result

    modified_plan = result.get("modified_plan")
    if not isinstance(modified_plan, dict):
        return result

    retrieved_items = fetch_retrieved_items_for_modify_nutrition_guard(request, user_feedback)
    guard_changes, guard_warnings, guard_recommendations, guard_sources = (
        apply_nutrition_safety_guard_to_modified_plan(
            modified_plan,
            user_feedback,
            retrieved_items,
        )
    )
    daily_meals = extract_nutrition_daily_meals(modified_plan)
    result["changes_summary"] = unique_strings(
        guard_changes,
        result.get("changes_summary"),
        limit=20,
    )
    result["recommendations"] = unique_strings(
        result.get("recommendations"),
        guard_recommendations,
        limit=8,
    )
    result["allergy_warnings"] = unique_strings(
        guard_warnings,
        result.get("allergy_warnings"),
    )
    result["sources"] = merge_sources_for_used_nutrition_foods(
        result.get("sources"),
        guard_sources,
        daily_meals,
        retrieved_items=retrieved_items,
    )
    if isinstance(modified_plan.get("total_daily"), dict):
        result["total_daily"] = modified_plan.get("total_daily")
    if isinstance(modified_plan.get("macro_targets"), dict):
        result["macro_targets"] = modified_plan.get("macro_targets")
    if modified_plan.get("daily_calorie_target") is not None:
        result["daily_calorie_target"] = modified_plan.get("daily_calorie_target")
    return result

def validate_modified_nutrition_sources(daily_meals, retrieved_items):
    retrieved_source_ids = {
        str(source_id)
        for item in retrieved_items or []
        for source_id in [retrieved_food_source_id(item)]
        if source_id is not None
    }
    for meal in daily_meals:
        for item in meal.get("items", []) or []:
            source_id = nutrition_source_id_from_item(item)
            if source_id is None or str(source_id) not in retrieved_source_ids:
                raise RAGGenerationError(
                    "rag_hallucinated_source_id",
                    f"Modified nutrition response used non-retrieved food source_id {source_id}",
                )

def modified_nutrition_rag_response_to_legacy_response(
    rag_response,
    request: SmartModifyNutritionRequest,
    user_feedback: Dict[str, Any],
    retrieved_items=None,
):
    daily_meals = []
    for meal in rag_response.modified_plan.daily_meals:
        items = [nutrition_food_to_legacy_item(food) for food in meal.foods]
        daily_meals.append({
            "meal": meal.meal,
            "items": items,
        })

    if not daily_meals:
        raise ValueError("LLM response did not include modified nutrition meals")
    if not any(meal["items"] for meal in daily_meals):
        raise ValueError("LLM response did not include usable modified nutrition foods")

    validate_modified_nutrition_sources(daily_meals, retrieved_items or [])

    modified_plan = copy.deepcopy(request.current_plan or {})
    if not isinstance(modified_plan, dict):
        modified_plan = {}
    modified_plan["daily_meals"] = daily_meals

    guard_changes, guard_warnings, guard_recommendations, guard_sources = (
        apply_nutrition_safety_guard_to_modified_plan(
            modified_plan,
            user_feedback,
            retrieved_items or [],
        )
    )
    daily_meals = extract_nutrition_daily_meals(modified_plan)
    total_daily = recalculate_nutrition_daily_totals(daily_meals)
    modified_plan["total_daily"] = total_daily

    normalized_target_macros = modified_nutrition_target_macros(request, user_feedback)
    estimated_target_calories = estimate_calories_from_macros(normalized_target_macros)
    if normalized_target_macros:
        modified_plan["macro_targets"] = normalized_target_macros
    elif rag_response.modified_plan.macro_targets:
        modified_plan["macro_targets"] = model_to_plain_dict(rag_response.modified_plan.macro_targets)
    if estimated_target_calories or rag_response.modified_plan.daily_calorie_target:
        modified_plan["daily_calorie_target"] = estimated_target_calories or rag_response.modified_plan.daily_calorie_target

    if estimated_target_calories and total_daily["calories"] < (estimated_target_calories * 0.55):
        raise RAGGenerationError(
            "nutrition_totals_too_low",
            (
                f"Modified nutrition totals too low: recalculated calories {total_daily['calories']:g} "
                f"below 55% of estimated target calories {estimated_target_calories:g}"
            ),
        )

    changes_summary = unique_strings(
        guard_changes,
        rag_response.changes_summary or [rag_response.summary],
        limit=20,
    )
    recommendations = unique_strings(
        rag_response.recommendations,
        guard_recommendations,
        limit=8,
    )
    allergy_warnings = unique_strings(
        guard_warnings,
        rag_response.allergy_warnings,
    )
    sources = merge_sources_for_used_nutrition_foods(
        rag_response.sources,
        guard_sources,
        daily_meals,
        retrieved_items=retrieved_items or [],
    )

    return {
        "status": rag_response.status,
        "plan_id": nutrition_plan_id_from_request(request),
        "version": next_nutrition_plan_version(request.current_plan),
        "changes_summary": changes_summary,
        "modified_plan": modified_plan,
        "recommendations": recommendations,
        "total_daily": total_daily,
        "daily_calorie_target": modified_plan.get("daily_calorie_target"),
        "macro_targets": modified_plan.get("macro_targets"),
        "generation_mode": "rag",
        "rag_summary": rag_response.summary,
        "sources": sources,
        "allergy_warnings": allergy_warnings,
    }

def generate_modified_nutrition_plan_rag(request: SmartModifyNutritionRequest):
    user_feedback = resolve_nutrition_modification_feedback(request)
    retrieval_query = build_modify_nutrition_rag_search_query(request, user_feedback)
    rag_logger.info("modify_nutrition_rag.retrieval_start user_id=%s", request.user_summary.user_id)

    try:
        results = search_nutrition(retrieval_query, n_results=30)
        retrieved_items = unpack_chroma_results(results)
    except Exception as error:
        log_nutrition_rag_exception("modify_nutrition_retrieval", error)
        raise RAGGenerationError("rag_generation_failed", sanitize_rag_error_message(error), error) from error

    rag_logger.info("modify_nutrition_rag.retrieval_done retrieved_foods=%s", len(retrieved_items))
    if not retrieved_items:
        message = "No retrieved foods found for RAG nutrition modification"
        rag_logger.warning("modify_nutrition_rag.retrieval_empty: %s", message)
        raise RAGGenerationError("no_retrieved_foods", message)

    prompt_results = deduplicate_food_results_for_prompt(results)
    deduped_items = unpack_chroma_results(prompt_results)
    if not deduped_items:
        message = "No retrieved foods found for RAG nutrition modification"
        rag_logger.warning("modify_nutrition_rag.dedup_empty: %s", message)
        raise RAGGenerationError("no_retrieved_foods", message)

    food_context = build_food_context(prompt_results)
    if not food_context.strip():
        message = "No retrieved foods found for RAG nutrition modification"
        rag_logger.warning("modify_nutrition_rag.context_empty: %s", message)
        raise RAGGenerationError("no_retrieved_foods", message)

    user_payload = build_modify_nutrition_user_payload(request, user_feedback, retrieval_query)
    messages = build_modify_nutrition_plan_prompt(user_payload, food_context)
    rag_logger.info(
        "modify_nutrition_rag.prompt_built message_count=%s context_chars=%s",
        len(messages),
        len(food_context),
    )

    try:
        rag_logger.info("modify_nutrition_rag.llm_call_start provider=%s", settings.LLM_PROVIDER)
        llm_text = call_llm_chat(
            messages,
            response_schema=modified_nutrition_plan_response_schema(),
        )
        rag_logger.info(
            "modify_nutrition_rag.llm_call_done provider=%s response_chars=%s",
            settings.LLM_PROVIDER,
            len(llm_text or ""),
        )
    except Exception as error:
        log_nutrition_rag_exception("modify_nutrition_llm_call", error)
        category = classify_llm_error(error)
        raise RAGGenerationError(category, sanitize_rag_error_message(error), error) from error

    try:
        rag_response = parse_modified_nutrition_plan_response(llm_text)
        rag_logger.info("modify_nutrition_rag.parse_done")
    except ValidationError as error:
        log_nutrition_rag_exception("modify_nutrition_parse_validation", error)
        raise RAGGenerationError(
            "rag_schema_validation_failed",
            sanitize_rag_error_message(error),
            error,
        ) from error
    except ValueError as error:
        log_nutrition_rag_exception("modify_nutrition_parse_json", error)
        raise RAGGenerationError("invalid_llm_json", sanitize_rag_error_message(error), error) from error
    except Exception as error:
        log_nutrition_rag_exception("modify_nutrition_parse_unknown", error)
        raise RAGGenerationError("invalid_llm_json", sanitize_rag_error_message(error), error) from error

    try:
        response = modified_nutrition_rag_response_to_legacy_response(
            rag_response,
            request,
            user_feedback,
            retrieved_items=deduped_items,
        )
        rag_logger.info("modify_nutrition_rag.legacy_conversion_done")
        return response
    except RAGGenerationError:
        raise
    except Exception as error:
        log_nutrition_rag_exception("modify_nutrition_legacy_conversion", error)
        raise RAGGenerationError(
            "rag_legacy_conversion_failed",
            sanitize_rag_error_message(error),
            error,
        ) from error


def analyze_nutrition_and_suggest_modifications(current_plan, user_feedback, search_func):
    modified_plan = copy.deepcopy(current_plan)
    changes_summary = []
    recommendations = []

    disliked_foods = [str(x).strip() for x in as_list(user_feedback.get("disliked_foods"))]
    liked_foods = [normalize_term(x) for x in as_list(user_feedback.get("liked_foods")) if normalize_term(x)]
    blocked_terms = collect_blocked_terms(user_feedback)

    goal = normalize_term(user_feedback.get("goal", "")).replace(" ", "_")
    notes = normalize_term(
        " ".join(
            str(item)
            for item in [
                user_feedback.get("notes", ""),
                user_feedback.get("modification_request", ""),
            ]
            if item
        )
    )

    if goal == "higher_protein":
        blocked_terms.update(
            expand_blocked_terms([
                "nuts",
                "almonds",
                "walnuts",
                "peanut",
                "peanut butter",
            ])
        )

    normalized_meals = []
    for meal in modified_plan.get("daily_meals", []):
        meal_name = meal.get("meal") or meal.get("meal_type")
        meal_items = meal.get("items", [])

        if not meal_items and meal.get("foods"):
            converted_items = []
            for food in meal.get("foods", []):
                converted_items.append({
                    "food_id": food.get("food_id") or food.get("nutrition_id"),
                    "name": food.get("name", ""),
                    "calories": float(food.get("calories", 0) or 0),
                    "protein": float(food.get("protein", 0) or 0),
                    "carbs": float(food.get("carbs", 0) or 0),
                    "fat": float(food.get("fat", 0) or 0),
                    "quantity": food.get("quantity", 1),
                })
            meal_items = converted_items

        normalized_meals.append({
            "meal": meal_name,
            "items": meal_items
        })

    modified_plan["daily_meals"] = normalized_meals

    search_queries = []

    if goal == "lower_calories":
        search_queries.extend([
            "low calorie high protein food",
            "healthy low calorie food",
            "lean protein food",
        ])
    elif goal == "higher_protein":
        search_queries.extend([
            "high protein lean food",
            "lean protein food",
            "healthy protein rich food",
        ])
    elif goal:
        search_queries.append(f"{goal} healthy food")

    for liked in liked_foods[:5]:
        search_queries.append(liked)

    if notes:
        search_queries.append(notes)

    if not search_queries:
        search_queries.append("healthy balanced food")

    suggested_foods = []
    seen_ids = set()
    seen_names = set()

    for query in search_queries[:10]:
        results = search_func(query, n_results=15)
        docs = results.get("documents", [[]])
        metas = results.get("metadatas", [[]])

        if not docs or not metas:
            continue

        for _, meta in zip(docs[0], metas[0]):
            food_id = meta.get("id")
            food_name = str(meta.get("name", "")).strip()
            food_name_l = normalize_term(food_name)

            if not food_id or not food_name:
                continue

            if food_id in seen_ids or food_name_l in seen_names:
                continue

            if food_matches_restrictions(food_name, blocked_terms):
                continue

            calories = float(meta.get("calories", 0) or 0)
            fat = float(meta.get("fat", 0) or 0)
            protein = float(meta.get("protein", 0) or 0)

            if goal == "lower_calories":
                if calories > 220:
                    continue
                if fat > 10:
                    continue

            if goal == "higher_protein":
                if protein < 12:
                    continue
                if fat > 12:
                    continue
                if calories > 250:
                    continue

            suggested_foods.append({
                "food_id": food_id,
                "source_id": food_id,
                "name": food_name,
                "calories": calories,
                "protein": protein,
                "carbs": float(meta.get("carbs", 0) or 0),
                "fat": fat,
                "quantity": 1
            })

            seen_ids.add(food_id)
            seen_names.add(food_name_l)

    suggested_foods.sort(key=lambda food: nutrition_candidate_rank(food, user_feedback))

    used_names = set()
    total_daily = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}

    for meal in modified_plan.get("daily_meals", []):
        meal_items = meal.get("items", [])

        for i, item in enumerate(meal_items):
            should_replace = False

            if item.get("name") and food_matches_restrictions(item.get("name", ""), blocked_terms):
                should_replace = True

            if goal == "lower_calories":
                try:
                    if float(item.get("calories", 0) or 0) > 220:
                        should_replace = True
                    if float(item.get("fat", 0) or 0) > 10:
                        should_replace = True
                except Exception:
                    pass

            if goal == "higher_protein":
                try:
                    if float(item.get("protein", 0) or 0) < 12:
                        should_replace = True
                    if float(item.get("fat", 0) or 0) > 12:
                        should_replace = True
                    if float(item.get("calories", 0) or 0) > 250:
                        should_replace = True
                except Exception:
                    pass

            if should_replace:
                replacement = None
                for food in suggested_foods:
                    candidate_name = normalize_term(food["name"])

                    if candidate_name in used_names:
                        continue

                    if food_matches_restrictions(food["name"], blocked_terms):
                        continue

                    replacement = food
                    break

                if replacement:
                    old_name = item.get("name", f"food_id {item.get('food_id', 'unknown')}")
                    meal_items[i] = replacement
                    used_names.add(normalize_term(replacement["name"]))
                    changes_summary.append(f"Replaced '{old_name}' with '{replacement['name']}'")

        meal["items"] = meal_items

    # final cleanup pass
    for meal in modified_plan.get("daily_meals", []):
        cleaned_items = []

        for item in meal.get("items", []):
            item_name = item.get("name", "")
            item_calories = float(item.get("calories", 0) or 0)
            item_fat = float(item.get("fat", 0) or 0)
            item_protein = float(item.get("protein", 0) or 0)

            if food_matches_restrictions(item_name, blocked_terms):
                continue

            if goal == "lower_calories":
                if item_calories > 220 or item_fat > 10:
                    continue

            if goal == "higher_protein":
                if item_protein < 12:
                    continue
                if item_fat > 12:
                    continue
                if item_calories > 250:
                    continue

            cleaned_items.append(item)

        meal["items"] = cleaned_items

    for meal in modified_plan.get("daily_meals", []):
        for item in meal.get("items", []):
            quantity = nutrition_item_quantity_number(item)
            total_daily["calories"] += float(item.get("calories", 0) or 0) * quantity
            total_daily["protein"] += float(item.get("protein", 0) or 0) * quantity
            total_daily["carbs"] += float(item.get("carbs", 0) or 0) * quantity
            total_daily["fat"] += float(item.get("fat", 0) or 0) * quantity

    modified_plan["total_daily"] = total_daily

    if goal == "lower_calories":
        recommendations.append("Prefer lower calorie meals with lean protein sources.")
    if goal == "higher_protein":
        recommendations.append("Increase lean protein sources while keeping calories and fats controlled.")
    if disliked_foods:
        recommendations.append("Removed or reduced disliked foods where alternatives were available.")
    if notes:
        recommendations.append("Applied the user's nutrition notes where possible.")
    if user_feedback.get("food_allergies") or user_feedback.get("allergies"):
        recommendations.append("Avoided foods that conflict with the user's allergy notes.")

    return {
        "plan_id": current_plan.get("plan_id", "unknown"),
        "version": int(current_plan.get("version", 1)) + 1,
        "changes_summary": changes_summary[:20],
        "modified_plan": modified_plan,
        "recommendations": recommendations[:6]
    }


@app.post("/modify-nutrition-plan")
async def modify_nutrition_plan(request: SmartModifyNutritionRequest):
    try:
        return generate_modified_nutrition_plan_rag(request)
    except Exception as rag_error:
        try:
            user_feedback = resolve_nutrition_modification_feedback(request)
            result = analyze_nutrition_and_suggest_modifications(
                current_plan=request.current_plan,
                user_feedback=user_feedback,
                search_func=search_nutrition
            )
            result = apply_nutrition_safety_guard_to_legacy_modify_result(
                result,
                request,
                user_feedback,
            )
            result["generation_mode"] = "fallback"
            result["fallback_reason"] = safe_nutrition_fallback_reason(rag_error)
            result.setdefault("status", "success")
            if settings.DEBUG:
                result["rag_debug_error_type"] = rag_debug_error_type(rag_error)
                result["rag_debug_error_message"] = sanitize_rag_error_message(rag_error)
            return result
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

def to_float_or_none(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def first_number(data: Dict[str, Any], keys: List[str], fallback=None):
    containers = [data]
    for parent_key in ["metrics", "summary", "latest", "progress"]:
        parent = data.get(parent_key)
        if isinstance(parent, dict):
            containers.append(parent)

    for container in containers:
        for key in keys:
            value = to_float_or_none(container.get(key))
            if value is not None:
                return value

    fallback_value = to_float_or_none(fallback)
    return fallback_value

def average_from_logs(progress_data: Dict[str, Any], log_key: str, value_keys: List[str]):
    logs = progress_data.get(log_key, [])
    if not isinstance(logs, list):
        return None

    values = []
    for item in logs:
        if not isinstance(item, dict):
            continue
        for key in value_keys:
            value = to_float_or_none(item.get(key))
            if value is not None:
                values.append(value)
                break

    if not values:
        return None
    return sum(values) / len(values)

def sequence_change(progress_data: Dict[str, Any], log_keys: List[str], value_keys: List[str], relative=False):
    for log_key in log_keys:
        logs = progress_data.get(log_key, [])
        if not isinstance(logs, list):
            continue

        values = []
        for item in logs:
            if isinstance(item, dict):
                for key in value_keys:
                    value = to_float_or_none(item.get(key))
                    if value is not None:
                        values.append(value)
                        break
            else:
                value = to_float_or_none(item)
                if value is not None:
                    values.append(value)

        if len(values) >= 2:
            change = values[-1] - values[0]
            if relative and values[0] != 0:
                return change / abs(values[0])
            return change

    return None

def build_progress_analysis(user_summary: UserSummary, progress_data: Dict[str, Any], current_plan_id=None):
    progress_data = progress_data or {}
    goal = normalize_term(user_summary.goal).replace(" ", "_")

    progress_rate = first_number(
        progress_data,
        ["progress_rate", "rate", "weekly_progress_rate"],
        user_summary.progress_rate,
    )
    consistency = first_number(
        progress_data,
        ["consistency_score", "consistency", "adherence_score"],
        user_summary.consistency_score,
    )

    completed_workouts = first_number(progress_data, ["completed_workouts", "workouts_completed"])
    planned_workouts = first_number(progress_data, ["planned_workouts", "scheduled_workouts"])
    workout_completion_rate = first_number(
        progress_data,
        ["workout_completion_rate", "training_adherence", "workout_adherence"],
    )
    if workout_completion_rate is None and planned_workouts and completed_workouts is not None:
        workout_completion_rate = completed_workouts / planned_workouts

    nutrition_adherence = first_number(
        progress_data,
        ["nutrition_adherence", "meal_adherence", "diet_adherence"],
    )
    if nutrition_adherence is None:
        nutrition_adherence = average_from_logs(
            progress_data,
            "nutrition_logs",
            ["adherence_score", "meal_adherence", "diet_adherence"],
        )

    weight_change = first_number(
        progress_data,
        ["weight_change", "weight_delta", "body_weight_change"],
    )
    if weight_change is None:
        weight_change = sequence_change(
            progress_data,
            ["weight_logs", "body_weight_logs", "weights"],
            ["weight", "body_weight", "value"],
        )

    strength_change = first_number(
        progress_data,
        ["strength_change", "strength_progress", "volume_change", "training_volume_change"],
    )
    if strength_change is None:
        strength_change = sequence_change(
            progress_data,
            ["workout_logs", "strength_logs", "exercise_logs"],
            ["total_volume", "volume", "estimated_1rm", "one_rep_max", "max_weight"],
            relative=True,
        )

    evidence = []
    metrics = {
        "progress_rate": progress_rate,
        "consistency_score": consistency,
        "workout_completion_rate": workout_completion_rate,
        "nutrition_adherence": nutrition_adherence,
        "weight_change": weight_change,
        "strength_change": strength_change,
    }

    for metric, value in metrics.items():
        if value is not None:
            evidence.append({"metric": metric, "value": round(value, 4)})

    reasons = []
    suggested_training_changes = []
    suggested_nutrition_changes = []

    if progress_rate is not None and progress_rate < 0.05:
        reasons.append("progress rate is low")
        suggested_training_changes.append("Review weekly volume, exercise selection, and progressive overload.")

    if consistency is not None and consistency < 0.6:
        reasons.append("overall consistency is low")
        suggested_training_changes.append("Reduce plan complexity or training frequency until adherence improves.")

    if workout_completion_rate is not None and workout_completion_rate < 0.6:
        reasons.append("workout completion rate is low")
        suggested_training_changes.append("Shorten sessions or reduce weekly training days to improve completion.")

    if strength_change is not None and strength_change < -0.05:
        reasons.append("strength or training volume is trending down")
        suggested_training_changes.append("Add a recovery-focused deload and avoid increasing load this week.")

    if nutrition_adherence is not None and nutrition_adherence < 0.6:
        reasons.append("nutrition adherence is low")
        suggested_nutrition_changes.append("Simplify meals and use easier high-protein staples.")

    if weight_change is not None:
        if goal in ["fat_loss", "weight_loss", "lower_calories"] and weight_change >= 0.2:
            reasons.append("weight is not moving toward the fat-loss goal")
            suggested_nutrition_changes.append("Recheck daily calories and increase lean protein/vegetable choices.")
        elif goal in ["muscle_gain", "higher_protein"] and weight_change <= -0.2:
            reasons.append("weight is dropping during a muscle-gain goal")
            suggested_nutrition_changes.append("Increase daily calories with controlled protein and carbohydrate portions.")

    if user_summary.injuries:
        suggested_training_changes.append("Keep exercise substitutions joint-friendly for active injuries.")

    needs_modification = bool(reasons)

    if not suggested_training_changes:
        suggested_training_changes.append("Keep the current training plan and monitor the next check-in.")
    if not suggested_nutrition_changes:
        suggested_nutrition_changes.append("Keep the current nutrition plan unless adherence drops.")

    if needs_modification:
        if any("adherence" in reason or "completion" in reason or "consistency" in reason for reason in reasons):
            status = "low_adherence"
        else:
            status = "plateau"
        summary = "Progress data suggests the plan may need adjustment."
        reason = "; ".join(reasons)
    elif progress_rate is not None and progress_rate > 0.15 and (consistency is None or consistency >= 0.75):
        status = "progressing_well"
        summary = "Progress data looks strong; no major change is needed."
        reason = "progress and consistency are acceptable"
    else:
        status = "steady_progress"
        summary = "Progress is acceptable; continue monitoring before making major changes."
        reason = "no strong negative trend detected"

    return {
        "user_id": user_summary.user_id,
        "current_plan_id": current_plan_id,
        "analysis_date": datetime.now().isoformat(),
        "status": status,
        "summary": summary,
        "needs_modification": needs_modification,
        "reason": reason,
        "evidence": evidence,
        "suggested_training_changes": suggested_training_changes[:6],
        "suggested_nutrition_changes": suggested_nutrition_changes[:6],
        # Backward-compatible fields for older callers.
        "metrics": metrics,
        "recommendations": (suggested_training_changes + suggested_nutrition_changes)[:6],
        "suggested_action": "modify_plan" if needs_modification else "continue",
    }

@app.post("/analyze-progress")
async def analyze_progress(request: AnalyzeProgressRequest):
    try:
        return build_progress_analysis(
            user_summary=request.user_summary,
            progress_data=request.progress_data,
            current_plan_id=request.current_plan_id,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# =====================
# 🚀 RUN
# =====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
