from pydantic import BaseModel, Field
from typing import List, Optional

class ExtractedPreferences(BaseModel):
    keywords: List[str] = Field(default=[], description="Extracted keywords from user text")
    negative_keywords: List[str] = Field(default=[], description="Keywords the user explicitly wants to avoid")
    target_meal: Optional[str] = Field(default=None, description="Specific meal target: Sarapan, Makan Siang, or Makan Malam")

class MacroTarget(BaseModel):
    calories: float = Field(..., example=1800.0)
    protein_g: float = Field(..., example=100.0)
    fat_g: float = Field(..., example=50.0)
    carb_g: float = Field(..., example=200.0)

class AllergyProfile(BaseModel):
    gluten: int = Field(0, example=0)
    dairy: int = Field(0, example=0)
    nuts: int = Field(0, example=0)
    peanut: int = Field(0, example=0)
    seafood: int = Field(0, example=1)
    egg: int = Field(0, example=1)
    soy: int = Field(0, example=0)
    celery: int = Field(0, example=0)

class MealPreferences(BaseModel):
    food_category: Optional[List[str]] = Field(default=[], example=["berkuah"])
    main_ingredients: Optional[List[str]] = Field(default=[], example=["ayam"])

class RecommendRequest(BaseModel):
    target_macros: MacroTarget
    allergies: AllergyProfile
    
    # Meal Specific Preferences (Optional)
    breakfast_prefs: Optional[MealPreferences] = None
    lunch_prefs: Optional[MealPreferences] = None
    dinner_prefs: Optional[MealPreferences] = None
    
    # Preference Method 2: Natural Language (Processed by Gen AI)
    user_text: Optional[str] = Field(default=None, example="Saya mau makan yang berkuah dan ada ayamnya di siang hari")

    # Optional planning controls
    start_date: Optional[str] = Field(default=None, example="2026-05-28")
    days: Optional[int] = Field(default=7, example=7)
    variety_penalty: Optional[float] = Field(default=0.15, example=0.15)

class FoodRecommendation(BaseModel):
    food_name: str
    calories_100g: float
    ideal_grams: float
    ideal_calories: float
    match_score: float

class MealRecommendations(BaseModel):
    meal_name: str
    target_calories: float
    recommendations: List[FoodRecommendation]

class RecommendResponse(BaseModel):
    daily_plan: List[MealRecommendations]
    narrative_summary: str
