# main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import os
import hashlib
import pymysql
import chromadb
import time
import json
import traceback
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import settings
import re
import copy

load_dotenv()

# =====================
# 🧠 EMBEDDINGS
# =====================
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2-preview",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# =====================
# 🗄️ MYSQL CONFIG
# =====================
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "user": os.getenv("DB_USERNAME", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_DATABASE", "gym"),
}

# =====================
# 🧠 CHROMA DB - Collections
# =====================
client = chromadb.PersistentClient(path="chroma_gym")
exercises_collection = client.get_or_create_collection(name="exercises_data")
nutrition_collection = client.get_or_create_collection(name="nutrition_data")

# =====================
# 📌 MANIFEST & CHECKPOINT
# =====================
EXERCISES_MANIFEST = "gym_manifest.json"
NUTRITION_MANIFEST = "nutrition_manifest.json"
CHECKPOINT_FILE = "./data/sync_checkpoint.json"

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

def fetch_changed_exercises(since_time):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            e.id,
            e.name,
            e.difficulty_level,
            e.instructions,
            e.common_mistakes,
            e.video_url,
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
       
    """)
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
# 🧾 TEXT FORMAT
# =====================
def exercise_row_to_text(row):
    return f"{row['name']} | {row['muscle_group']} | {row['difficulty_level']}"

def nutrition_row_to_text(row):
    return f"{row['name']} | {row['category_name']} | {row['calories']} cal"

# =====================
# 🔐 HASH FUNCTION
# =====================
def hash_row(text: str):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

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
    return embeddings.embed_query(text)

# =====================
# 📦 PROCESS BATCH (عام)
# =====================
def process_batch(rows, collection, manifest_file, row_to_text_func, get_metadata_func, batch_num, total_batches, data_type):
    print(f"  📦 Processing {data_type} batch {batch_num}/{total_batches} ({len(rows)} items)...")
    
    old_manifest = load_manifest(manifest_file)
    new_manifest = dict(old_manifest)
    added, updated = 0, 0
    
    for row in rows:
        embedding_text = row['embedding_text']
        hash_text = row_to_text_func(row)
        row_hash = hash_row(hash_text)
        row_id = str(row["id"])
        
        new_manifest[row_id] = row_hash
        old_hash = old_manifest.get(row_id)
        
        metadata = get_metadata_func(row)
        
        if old_hash is None:
            vector = embed(embedding_text)
            collection.add(
                ids=[row_id],
                documents=[embedding_text],
                embeddings=[vector],
                metadatas=[metadata]
            )
            added += 1
        elif old_hash != row_hash:
            vector = embed(embedding_text)
            collection.update(
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
    return {
        "id": row["id"],
        "name": row["name"],
        "muscle_group": row["muscle_group"],
        "difficulty": row["difficulty_level"],
        "type": "exercise"
    }

def get_nutrition_metadata(row):
    return {
        "id": row["id"],
        "name": row["name"],
        "category": row["category_name"],
        "calories": float(row["calories"]) if row.get("calories") else 0,
        "protein": float(row["protein"]) if row.get("protein") else 0,
        "carbs": float(row["carbs"]) if row.get("carbs") else 0,
        "fat": float(row["fat"]) if row.get("fat") else 0,
        "type": "food"
    }

# =====================
# 🔄 SYNC FUNCTIONS
# =====================
def sync_exercises_to_vector(full_sync=False):
    print("🔄 Syncing Exercises...")
    total_added, total_updated = 0, 0
    
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
                    batch_num + 1, total_batches, "exercises"
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
                    batch_num + 1, total_batches, "nutrition"
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

def sync_all(full_sync=False):
    print("🔄 Starting Full Smart Sync...")
    start_time = time.time()
    
    exercises_added, exercises_updated = sync_exercises_to_vector(full_sync)
    nutrition_added, nutrition_updated = sync_nutrition_to_vector(full_sync)
    
    current_exercise_ids = set(fetch_all_exercise_ids())
    old_exercise_manifest = load_manifest(EXERCISES_MANIFEST)
    deleted_exercises = set(old_exercise_manifest.keys()) - current_exercise_ids
    for row_id in deleted_exercises:
        exercises_collection.delete(ids=[row_id])
    
    current_nutrition_ids = set(fetch_all_nutrition_ids())
    old_nutrition_manifest = load_manifest(NUTRITION_MANIFEST)
    deleted_nutrition = set(old_nutrition_manifest.keys()) - current_nutrition_ids
    for row_id in deleted_nutrition:
        nutrition_collection.delete(ids=[row_id])
    
    save_checkpoint({
        "last_sync_time": datetime.now().isoformat(),
        "exercises_batch": 0,
        "nutrition_batch": 0
    })
    
    elapsed_time = time.time() - start_time
    print(f"\n✅ ALL SYNC DONE in {elapsed_time:.2f} seconds")
    print(f"   Exercises: +{exercises_added} added, 🔄{exercises_updated} updated, 🗑{len(deleted_exercises)} deleted")
    print(f"   Nutrition: +{nutrition_added} added, 🔄{nutrition_updated} updated, 🗑{len(deleted_nutrition)} deleted")
    
    return {
        "exercises": {"added": exercises_added, "updated": exercises_updated, "deleted": len(deleted_exercises)},
        "nutrition": {"added": nutrition_added, "updated": nutrition_updated, "deleted": len(deleted_nutrition)},
        "elapsed_seconds": elapsed_time
    }

# =====================
# 🔍 SEARCH FUNCTIONS
# =====================
def search_exercises(query, n_results=10):
    query_vector = embed(query)
    results = exercises_collection.query(
        query_embeddings=[query_vector],
        n_results=n_results
    )
    return results

def search_nutrition(query, n_results=10):
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
    current_plan_id: Optional[str] = None

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
            print("DEBUG_MIXED_FOCUS", focus, ex_name, exercise_matches_focus(ex_name, focus))

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
app = FastAPI(title="Gym AI Service", debug=settings.DEBUG)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

@app.post("/sync-all")
async def sync_all_data(full_sync: bool = Query(False)):
    try:
        result = sync_all(full_sync=full_sync)
        return {"status": "success", "stats": result}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search-exercises")
async def search_exercises_api(query: str, n_results: int = 10):
    try:
        results = search_exercises(query, n_results)
        return {
            "query": query,
            "type": "exercises",
            "results": [
                {"document": doc, "metadata": meta}
                for doc, meta in zip(results["documents"][0], results["metadatas"][0])
            ]
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search-foods")
async def search_foods_api(query: str, n_results: int = 10):
    try:
        results = search_nutrition(query, n_results)
        return {
            "query": query,
            "type": "foods",
            "results": [
                {"document": doc, "metadata": meta}
                for doc, meta in zip(results["documents"][0], results["metadatas"][0])
            ]
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-training-plan")
async def generate_training_plan(request: GenerateTrainingRequest):
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
    recommendations.append(f"DEBUG_GOAL={goal}")
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

    recommendations.append("FINAL_CLEANUP_OK")

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

@app.post("/analyze-progress")
async def analyze_progress(request: AnalyzeProgressRequest):
    try:
        progress_rate = request.user_summary.progress_rate
        consistency = request.user_summary.consistency_score
        
        if progress_rate < 0.05:
            status = "plateau"
            recommendation = "Increase volume or change exercises"
            action = "modify_plan"
        elif progress_rate > 0.15:
            status = "progressing_well"
            recommendation = "Continue current plan"
            action = "continue"
        else:
            status = "steady_progress"
            recommendation = "Minor adjustments recommended"
            action = "moderate"
        
        return {
            "user_id": request.user_summary.user_id,
            "analysis_date": datetime.now().isoformat(),
            "status": status,
            "metrics": {
                "progress_rate": progress_rate,
                "consistency_score": consistency,
                "estimated_next_plateau_days": 14 if progress_rate > 0.1 else 7
            },
            "recommendations": [recommendation],
            "suggested_action": action
        }
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