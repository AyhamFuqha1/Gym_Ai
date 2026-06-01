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
from llm_client import call_openrouter_chat
from prompt_builders import build_training_plan_prompt
from response_parsers import parse_training_plan_response
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
    current_plan_id: str
    current_plan: Dict[str, Any]
    user_summary: UserSummary
    user_feedback: Dict[str, Any] = {}

class SmartModifyNutritionRequest(BaseModel):
    current_plan_id: str
    current_plan: Dict[str, Any]
    user_summary: UserSummary
    user_feedback: Dict[str, Any] = {}

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

def classify_openrouter_error(error: Exception):
    message = str(error or "")
    if "OPENROUTER_API_KEY" in message or "OPENROUTER_MODEL" in message:
        return "openrouter_not_configured"
    if "timed out" in message.lower() or "timeout" in type(error).__name__.lower():
        return "openrouter_timeout"
    if "status " in message and "OpenRouter request failed" in message:
        return "openrouter_http_error"
    if "empty assistant message content" in message or "did not contain assistant message content" in message:
        return "llm_empty_response"
    return "rag_generation_failed"

def safe_training_fallback_reason(error: Exception):
    if isinstance(error, RAGGenerationError):
        return error.category

    message = str(error or "")
    if "OPENROUTER_API_KEY" in message or "OPENROUTER_MODEL" in message:
        return "openrouter_not_configured"
    if "timed out" in message.lower():
        return "openrouter_timeout"
    if "OpenRouter request failed with status" in message:
        return "openrouter_http_error"
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

def training_rag_response_to_legacy_response(rag_response, request: GenerateTrainingRequest):
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
        "sources": [model_to_plain_dict(source) for source in rag_response.sources],
        "injury_warnings": rag_response.injury_warnings,
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
        rag_logger.info("training_rag.openrouter_call_start")
        llm_text = call_openrouter_chat(messages)
        rag_logger.info("training_rag.openrouter_call_done response_chars=%s", len(llm_text or ""))
    except Exception as error:
        log_rag_exception("openrouter_call", error)
        category = classify_openrouter_error(error)
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
        response = training_rag_response_to_legacy_response(rag_response, request)
        rag_logger.info("training_rag.legacy_conversion_done")
        return response
    except Exception as error:
        log_rag_exception("legacy_conversion", error)
        raise RAGGenerationError(
            "rag_legacy_conversion_failed",
            sanitize_rag_error_message(error),
            error,
        ) from error

@app.post("/generate-training-plan")
async def generate_training_plan(request: GenerateTrainingRequest):
    try:
        return generate_training_plan_rag(request)
    except Exception as rag_error:
        fallback = generate_training_plan_rule_based(request)
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
        result = analyze_plan_and_suggest_modifications(
            current_plan=request.current_plan,
            user_feedback=request.user_feedback,
            search_func=search_exercises
        )
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


@app.post("/generate-nutrition-plan")
async def generate_nutrition_plan(request: GenerateNutritionRequest):
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


def analyze_nutrition_and_suggest_modifications(current_plan, user_feedback, search_func):
    modified_plan = copy.deepcopy(current_plan)
    changes_summary = []
    recommendations = []

    disliked_foods = [str(x).strip() for x in user_feedback.get("disliked_foods", [])]
    liked_foods = [normalize_term(x) for x in user_feedback.get("liked_foods", []) if normalize_term(x)]
    blocked_terms = expand_blocked_terms(disliked_foods)

    goal = normalize_term(user_feedback.get("goal", "")).replace(" ", "_")
    notes = normalize_term(user_feedback.get("notes", ""))

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
                "name": food_name,
                "calories": calories,
                "protein": protein,
                "carbs": float(meta.get("carbs", 0) or 0),
                "fat": fat,
                "quantity": 1
            })

            seen_ids.add(food_id)
            seen_names.add(food_name_l)

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
            quantity = float(item.get("quantity", 1) or 1)
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
        result = analyze_nutrition_and_suggest_modifications(
            current_plan=request.current_plan,
            user_feedback=request.user_feedback,
            search_func=search_nutrition
        )
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
