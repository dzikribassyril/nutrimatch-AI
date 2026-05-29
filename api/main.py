from fastapi import FastAPI, HTTPException
from schemas import RecommendRequest, RecommendResponse, FoodRecommendation
import data_service
import ml_model
import genai_service

app = FastAPI(
    title="NutriMatch AI API",
    description="Food Recommendation API combining Deep Learning and Gen AI",
    version="1.0.1"
)

@app.get("/")
def read_root():
    return {"message": "Welcome to NutriMatch AI API"}

@app.post("/api/recommend", response_model=RecommendResponse)
def recommend_food(req: RecommendRequest):
    extracted_keywords = []
    negative_keywords = []
    target_meal_from_ai = None
    
    if req.user_text:
        extracted_prefs = genai_service.extract_food_preference(req.user_text)
        extracted_keywords = extracted_prefs.keywords
        negative_keywords = extracted_prefs.negative_keywords
        target_meal_from_ai = extracted_prefs.target_meal
        
    print(f"Gen AI Extracted: Keywords={extracted_keywords}, Negatives={negative_keywords}, Target Meal={target_meal_from_ai}")
    
    food_df = data_service.get_food_data()
    if food_df.empty:
        raise HTTPException(status_code=500, detail="Database is empty or not loaded.")
    
    user_max = data_service.get_user_max()
    macro_array = [req.target_macros.calories, req.target_macros.protein_g, req.target_macros.fat_g, req.target_macros.carb_g]
    allergy_array = [
        req.allergies.gluten, req.allergies.dairy, req.allergies.nuts, req.allergies.peanut, 
        req.allergies.seafood, req.allergies.egg, req.allergies.soy, req.allergies.celery
    ]
    
    ratios = [
        ("breakfast", "Sarapan (25%)", 0.25, req.breakfast_prefs),
        ("lunch", "Makan Siang (40%)", 0.40, req.lunch_prefs),
        ("dinner", "Makan Malam (35%)", 0.35, req.dinner_prefs),
    ]

    meal_datasets = {}
    for meal_key, meal_label, ratio, prefs in ratios:
        cats = prefs.food_category if prefs else []
        ings = prefs.main_ingredients if prefs else []

        apply_ai_keywords = True
        if target_meal_from_ai and target_meal_from_ai.lower() not in meal_label.lower():
            apply_ai_keywords = False

        if apply_ai_keywords:
            ings = ings + extracted_keywords

        clean_meal_name = meal_label.split(' (')[0]
        meal_specific_df = data_service.filter_foods_by_preferences(
            categories=cats, ingredients=ings, target_meal=clean_meal_name,
            dataset=food_df, negative_keywords=negative_keywords if apply_ai_keywords else [],
        )

        meal_datasets[meal_key] = meal_specific_df

    daily_plan_raw = ml_model.generate_7day_combo_plan(
        daily_macros=macro_array, user_allergy_list=allergy_array,
        meal_datasets=meal_datasets, user_max=user_max,
        variety_penalty=req.variety_penalty if req.variety_penalty is not None else 0.20,
        start_date=req.start_date, days=req.days if req.days is not None else 7,
    )

    daily_plan_response = []
    for meal in daily_plan_raw:
        formatted_recs = [
            FoodRecommendation(
                food_name=r['food_name'],
                calories_100g=r['calories_100g'],
                ideal_grams=r['ideal_grams'],
                ideal_calories=r['ideal_calories'],
                match_score=r['match_score'],
            )
            for r in meal['recommendations']
        ]
        daily_plan_response.append({
            "meal_name": meal["meal_name"],
            "target_calories": meal["target_calories"],
            "recommendations": formatted_recs,
        })

    has_any_recs = any(meal["recommendations"] for meal in daily_plan_response)
    if not has_any_recs:
        return RecommendResponse(
            daily_plan=[],
            narrative_summary="Maaf, kami tidak menemukan makanan yang aman dengan preferensi dan alergi Anda.",
        )

    # Mengambil sampel makanan paling relevan untuk diolah narasi oleh Gen AI
    top_food_sample = None
    for meal in daily_plan_response:
        if "Makan Siang" in meal["meal_name"] and meal["recommendations"]:
            top_food_sample = meal["recommendations"][0]
            break

    if top_food_sample is None:
        for meal in daily_plan_response:
            if meal["recommendations"]:
                top_food_sample = meal["recommendations"][0]
                break

    # Hilangkan prefix tag [Makanan Utama]/[Karbo] saat dikirim ke GenAI
    clean_food_name = top_food_sample.food_name.split("] ")[-1] if "] " in top_food_sample.food_name else top_food_sample.food_name

    narrative = genai_service.generate_narrative(
        user_text=req.user_text,
        recommended_food=clean_food_name,
        portion=top_food_sample.ideal_grams,
        target_cal=req.target_macros.calories,
    )

    return RecommendResponse(
        daily_plan=daily_plan_response,
        narrative_summary=narrative,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)