from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


FlexibleId = Union[int, str]


class RAGSource(BaseModel):
    model_config = ConfigDict(extra="allow")

    source_id: FlexibleId
    source_table: str
    source_name: str
    score: Optional[float] = None
    reason_used: str


class TrainingExerciseRAGItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    exercise_id: FlexibleId
    name: str
    sets: Union[int, str]
    reps: Union[int, str]
    rest_seconds: Optional[int] = None
    intensity: Optional[str] = None
    reason: str
    source_id: FlexibleId


class TrainingDayRAGItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    day: Union[int, str]
    focus: str
    exercises: List[TrainingExerciseRAGItem] = Field(default_factory=list)


class TrainingPlanRAGResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    plan_type: str
    user_id: FlexibleId
    summary: str
    days: List[TrainingDayRAGItem] = Field(default_factory=list)
    injury_warnings: List[str] = Field(default_factory=list)
    sources: List[RAGSource] = Field(default_factory=list)


class MacroTargets(BaseModel):
    model_config = ConfigDict(extra="allow")

    protein_g: float
    carbs_g: float
    fat_g: float


class NutritionFoodRAGItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    food_id: FlexibleId
    name: str
    serving_size: Optional[str] = None
    quantity: Union[float, int, str] = 1
    calories: float
    protein: float
    carbs: float
    fat: float
    reason: str
    source_id: FlexibleId


class NutritionMealRAGItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    meal: str
    foods: List[NutritionFoodRAGItem] = Field(default_factory=list)


class NutritionPlanRAGResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    plan_type: str
    user_id: FlexibleId
    daily_calorie_target: Optional[float] = None
    macro_targets: MacroTargets
    meals: List[NutritionMealRAGItem] = Field(default_factory=list)
    allergy_warnings: List[str] = Field(default_factory=list)
    sources: List[RAGSource] = Field(default_factory=list)


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    return model.model_dump()


def _schema_type(schema_type: str, **extra: Any) -> Dict[str, Any]:
    schema = {"type": schema_type}
    schema.update(extra)
    return schema


def _source_schema(source_table: str) -> Dict[str, Any]:
    return _schema_type(
        "OBJECT",
        properties={
            "source_id": _schema_type("INTEGER"),
            "source_table": _schema_type("STRING", description=f"Use '{source_table}'."),
            "source_name": _schema_type("STRING"),
            "score": _schema_type("NUMBER"),
            "reason_used": _schema_type("STRING"),
        },
        required=["source_id", "source_table", "source_name", "reason_used"],
        propertyOrdering=["source_id", "source_table", "source_name", "score", "reason_used"],
    )


def training_plan_response_schema() -> Dict[str, Any]:
    """Gemini-compatible schema for TrainingPlanRAGResponse.

    This intentionally avoids Pydantic JSON Schema features such as $defs,
    anyOf, defaults, and titles because Gemini responseSchema supports a
    smaller OpenAPI-style schema subset.
    """
    exercise_schema = _schema_type(
        "OBJECT",
        properties={
            "exercise_id": _schema_type("INTEGER"),
            "name": _schema_type("STRING"),
            "sets": _schema_type("INTEGER"),
            "reps": _schema_type("STRING"),
            "rest_seconds": _schema_type("INTEGER"),
            "intensity": _schema_type("STRING"),
            "reason": _schema_type("STRING"),
            "source_id": _schema_type("INTEGER"),
        },
        required=["exercise_id", "name", "sets", "reps", "reason", "source_id"],
        propertyOrdering=[
            "exercise_id",
            "name",
            "sets",
            "reps",
            "rest_seconds",
            "intensity",
            "reason",
            "source_id",
        ],
    )
    day_schema = _schema_type(
        "OBJECT",
        properties={
            "day": _schema_type("INTEGER"),
            "focus": _schema_type("STRING"),
            "exercises": _schema_type("ARRAY", items=exercise_schema),
        },
        required=["day", "focus", "exercises"],
        propertyOrdering=["day", "focus", "exercises"],
    )
    return _schema_type(
        "OBJECT",
        properties={
            "status": _schema_type("STRING"),
            "plan_type": _schema_type("STRING", description="Use 'training'."),
            "user_id": _schema_type("INTEGER"),
            "summary": _schema_type("STRING"),
            "days": _schema_type("ARRAY", items=day_schema),
            "injury_warnings": _schema_type("ARRAY", items=_schema_type("STRING")),
            "sources": _schema_type("ARRAY", items=_source_schema("exercises")),
        },
        required=[
            "status",
            "plan_type",
            "user_id",
            "summary",
            "days",
            "sources",
            "injury_warnings",
        ],
        propertyOrdering=[
            "status",
            "plan_type",
            "user_id",
            "summary",
            "days",
            "injury_warnings",
            "sources",
        ],
    )


def nutrition_plan_response_schema() -> Dict[str, Any]:
    """Gemini-compatible schema for NutritionPlanRAGResponse."""
    macro_schema = _schema_type(
        "OBJECT",
        properties={
            "protein_g": _schema_type("NUMBER"),
            "carbs_g": _schema_type("NUMBER"),
            "fat_g": _schema_type("NUMBER"),
        },
        required=["protein_g", "carbs_g", "fat_g"],
        propertyOrdering=["protein_g", "carbs_g", "fat_g"],
    )
    food_schema = _schema_type(
        "OBJECT",
        properties={
            "food_id": _schema_type("INTEGER"),
            "name": _schema_type("STRING"),
            "serving_size": _schema_type("STRING"),
            "quantity": _schema_type("NUMBER"),
            "calories": _schema_type("NUMBER"),
            "protein": _schema_type("NUMBER"),
            "carbs": _schema_type("NUMBER"),
            "fat": _schema_type("NUMBER"),
            "reason": _schema_type("STRING"),
            "source_id": _schema_type("INTEGER"),
        },
        required=[
            "food_id",
            "name",
            "quantity",
            "calories",
            "protein",
            "carbs",
            "fat",
            "reason",
            "source_id",
        ],
        propertyOrdering=[
            "food_id",
            "name",
            "serving_size",
            "quantity",
            "calories",
            "protein",
            "carbs",
            "fat",
            "reason",
            "source_id",
        ],
    )
    meal_schema = _schema_type(
        "OBJECT",
        properties={
            "meal": _schema_type("STRING"),
            "foods": _schema_type("ARRAY", items=food_schema),
        },
        required=["meal", "foods"],
        propertyOrdering=["meal", "foods"],
    )
    return _schema_type(
        "OBJECT",
        properties={
            "status": _schema_type("STRING"),
            "plan_type": _schema_type("STRING", description="Use 'nutrition'."),
            "user_id": _schema_type("INTEGER"),
            "daily_calorie_target": _schema_type("NUMBER"),
            "macro_targets": macro_schema,
            "meals": _schema_type("ARRAY", items=meal_schema),
            "allergy_warnings": _schema_type("ARRAY", items=_schema_type("STRING")),
            "sources": _schema_type("ARRAY", items=_source_schema("foods")),
        },
        required=[
            "status",
            "plan_type",
            "user_id",
            "daily_calorie_target",
            "macro_targets",
            "meals",
            "sources",
            "allergy_warnings",
        ],
        propertyOrdering=[
            "status",
            "plan_type",
            "user_id",
            "daily_calorie_target",
            "macro_targets",
            "meals",
            "allergy_warnings",
            "sources",
        ],
    )
