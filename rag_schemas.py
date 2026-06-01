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
