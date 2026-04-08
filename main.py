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
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import settings

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
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "gym",
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
def analyze_plan_and_suggest_modifications(current_plan, user_feedback, search_func):
    modified_plan = current_plan.copy()
    changes_summary = []
    recommendations = []
    
    pain_areas = user_feedback.get('pain_areas', [])
    difficulty = user_feedback.get('difficulty', '')
    disliked_exercises = user_feedback.get('disliked_exercises', [])
    liked_exercises = user_feedback.get('liked_exercises', [])
    
    search_queries = []
    
    if pain_areas:
        for pain in pain_areas:
            search_queries.append(f"safe exercise for {pain} no pain alternative")
    
    if difficulty == "too_hard":
        search_queries.append("beginner easier exercise")
    elif difficulty == "too_easy":
        search_queries.append("advanced challenging exercise")
    
    if not search_queries:
        search_queries.append("standard alternative exercise")
    
    suggested_exercises = []
    for query in search_queries[:3]:
        results = search_func(query, n_results=10)
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            if meta["id"] not in [e.get("exercise_id") for e in suggested_exercises]:
                suggested_exercises.append({
                    "id": meta["id"],
                    "name": meta["name"],
                    "muscle_group": meta.get("muscle_group", ""),
                    "difficulty": meta.get("difficulty", "intermediate")
                })
    
    if "schedule" in modified_plan.get("plan_data", {}):
        for day_idx, day in enumerate(modified_plan["plan_data"]["schedule"]):
            for exercise_idx, exercise in enumerate(day.get("exercises", [])):
                exercise_id = exercise.get("exercise_id")
                
                if exercise_id in disliked_exercises:
                    if suggested_exercises:
                        new_exercise = suggested_exercises.pop(0)
                        old_name = exercise.get("name", "Unknown")
                        exercise.update({
                            "exercise_id": new_exercise["id"],
                            "name": new_exercise["name"],
                            "muscle_group": new_exercise["muscle_group"],
                            "difficulty": new_exercise["difficulty"]
                        })
                        changes_summary.append(f"Replaced '{old_name}' with '{new_exercise['name']}'")
                
                if difficulty == "too_hard":
                    if exercise.get("sets", 3) > 3:
                        exercise["sets"] = max(2, exercise.get("sets", 3) - 1)
                        changes_summary.append(f"Reduced sets for {exercise.get('name')}")
                elif difficulty == "too_easy":
                    if exercise.get("sets", 3) < 5:
                        exercise["sets"] = exercise.get("sets", 3) + 1
                        changes_summary.append(f"Increased sets for {exercise.get('name')}")
    
    if pain_areas:
        recommendations.append(f"Avoid exercises that strain the {', '.join(pain_areas)}. Focus on proper form.")
    if difficulty == "too_hard":
        recommendations.append("Consider reducing weight or taking longer rest periods between sets.")
    elif difficulty == "too_easy":
        recommendations.append("Try increasing weight gradually or adding more volume.")
    
    recommendations.append("Focus on mind-muscle connection and proper form.")
    
    return {
        "plan_id": current_plan.get("plan_id", "unknown"),
       "version": int(current_plan.get("version", 1)) + 1,
        "changes_summary": changes_summary[:10],
        "modified_plan": modified_plan,
        "recommendations": recommendations[:5]
    }

def analyze_nutrition_and_suggest_modifications(current_plan, user_feedback, search_func):
    modified_plan = current_plan.copy()
    changes_summary = []
    recommendations = []
    
    feedback = user_feedback
    current_macros = current_plan.get("total_daily", {})
    current_calories = current_macros.get("calories", 2000)
    
    search_queries = []
    
    if feedback.get("satiety") == "hungry_between_meals":
        search_queries.append("high protein high fiber filling snack")
        recommendations.append("Add protein-rich snacks between meals to stay full longer")
    
    if feedback.get("digestion_issues"):
        search_queries.append("easy digest light meal low fat")
        recommendations.append("Avoid heavy fried foods, eat smaller portions more frequently")
    
    if feedback.get("goal") == "muscle_gain":
        if current_macros.get("protein", 0) < 1.6 * 80:
            search_queries.append("high protein food for muscle building")
            recommendations.append("Increase protein intake to support muscle growth")
    
    if feedback.get("goal") == "fat_loss":
        if current_calories > 2000:
            search_queries.append("low calorie filling food")
            recommendations.append("Consider reducing calorie intake or increasing activity")
    
    if not search_queries:
        search_queries.append("healthy balanced meal")
    
    suggested_foods = []
    for query in search_queries[:3]:
        results = search_func(query, n_results=5)
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            if meta["id"] not in [f.get("food_id") for f in suggested_foods]:
                suggested_foods.append({
                    "food_id": meta["id"],
                    "name": meta["name"],
                    "calories": meta["calories"],
                    "protein": meta["protein"],
                    "carbs": meta["carbs"],
                    "fat": meta["fat"]
                })
    
    if "daily_meals" in modified_plan:
        for meal_idx, meal in enumerate(modified_plan["daily_meals"]):
            for item_idx, item in enumerate(meal.get("items", [])):
                if suggested_foods and (feedback.get("satiety") or feedback.get("digestion_issues")):
                    new_food = suggested_foods.pop(0)
                    old_name = item.get("name", "Unknown")
                    item.update({
                        "food_id": new_food["food_id"],
                        "name": new_food["name"],
                        "calories": new_food["calories"],
                        "protein": new_food["protein"],
                        "carbs": new_food["carbs"],
                        "fat": new_food["fat"]
                    })
                    changes_summary.append(f"Replaced '{old_name}' with '{new_food['name']}' in {meal.get('meal', 'meal')}")
                    break
    
    total_calories = 0
    total_protein = 0
    total_carbs = 0
    total_fat = 0
    
    for meal in modified_plan.get("daily_meals", []):
        for item in meal.get("items", []):
            total_calories += item.get("calories", 0)
            total_protein += item.get("protein", 0)
            total_carbs += item.get("carbs", 0)
            total_fat += item.get("fat", 0)
    
    modified_plan["total_daily"] = {
        "calories": total_calories,
        "protein": total_protein,
        "carbs": total_carbs,
        "fat": total_fat
    }
    
    return {
        "plan_id": current_plan.get("plan_id", "unknown"),
       "version": int(current_plan.get("version", 1)) + 1,
        "changes_summary": changes_summary[:10],
        "modified_plan": modified_plan,
        "recommendations": recommendations[:5]
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
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-training-plan")
async def generate_training_plan(request: GenerateTrainingRequest):
    try:
        search_query = f"{request.user_summary.goal} {request.user_summary.level} workout"
        if request.user_summary.weak_points:
            search_query += f" focus on {', '.join(request.user_summary.weak_points)}"
        
        results = search_exercises(search_query, n_results=20)
        
        schedule = []
        day_exercises = []
        for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            day_exercises.append({
                "exercise_id": meta["id"],
                "name": meta["name"],
                "muscle_group": meta["muscle_group"],
                "difficulty": meta["difficulty"],
                "sets": 3,
                "reps": "8-12" if meta["difficulty"] == "beginner" else "8-10",
                "rest_seconds": 90
            })
            if len(day_exercises) >= 6:
                schedule.append({"day": len(schedule) + 1, "focus": "Workout", "exercises": day_exercises})
                day_exercises = []
        
        if day_exercises:
            schedule.append({"day": len(schedule) + 1, "focus": "Workout", "exercises": day_exercises})
        
        return {
            "plan_id": f"plan_{request.user_summary.user_id}_{int(datetime.now().timestamp())}",
            "version": 1,
            "generated_at": datetime.now().isoformat(),
            "plan_data": {
                "duration_weeks": 4,
                "schedule": schedule[:5]
            }
        }
    except Exception as e:
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
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-nutrition-plan")
async def generate_nutrition_plan(request: GenerateNutritionRequest):
    try:
        search_query = f"{request.user_summary.goal} healthy food"
        if request.user_summary.goal == "muscle_gain":
            search_query += " high protein"
        elif request.user_summary.goal == "fat_loss":
            search_query += " low calorie"
        
        results = search_nutrition(search_query, n_results=15)
        
        meals = {"breakfast": [], "lunch": [], "dinner": [], "snacks": []}
        categories = ["breakfast", "lunch", "dinner", "snacks"]
        cat_idx = 0
        
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            meals[categories[cat_idx % 4]].append({
                "food_id": meta["id"],
                "name": meta["name"],
                "calories": meta["calories"],
                "protein": meta["protein"],
                "carbs": meta["carbs"],
                "fat": meta["fat"],
                "quantity": 1
            })
            cat_idx += 1
        
        total_calories = sum(item["calories"] for meal in meals.values() for item in meal if "calories" in item)
        
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
        raise HTTPException(status_code=500, detail=str(e))

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