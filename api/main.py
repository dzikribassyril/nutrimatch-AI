from fastapi import FastAPI, HTTPException
from schemas import RecommendRequest, RecommendResponse, FoodRecommendation
import data_service
import ml_model
import genai_service

app = FastAPI(
    title="NutriMatch AI API",
    description="Food Recommendation API combining Deep Learning and Gen AI",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"message": "Welcome to NutriMatch AI API"}

@app.post("/api/recommend", response_model=RecommendResponse)
def recommend_food(req: RecommendRequest):
    # 1. Gen AI Parsing (if user_text is provided)
    extracted_keywords = []
    negative_keywords = []
    target_meal_from_ai = None
    if req.user_text:
        extracted_prefs = genai_service.extract_food_preference(req.user_text)
        extracted_keywords = extracted_prefs.keywords
        negative_keywords = extracted_prefs.negative_keywords
        target_meal_from_ai = extracted_prefs.target_meal
        
    print(f"Gen AI Extracted: Keywords={extracted_keywords}, Negatives={negative_keywords}, Target Meal={target_meal_from_ai}")
    
    # 2. Check Database
    food_df = data_service.get_food_data()
    if food_df.empty:
        raise HTTPException(status_code=500, detail="Database is empty or not loaded.")
    
    # 3. TensorFlow Recommendation prep
    user_max = data_service.get_user_max()
    
    # Konversi Object ke Array berurutan agar bisa dibaca model AI
    macro_array = [
        req.target_macros.calories, 
        req.target_macros.protein_g, 
        req.target_macros.fat_g, 
        req.target_macros.carb_g
    ]
    
    allergy_array = [
        req.allergies.gluten, 
        req.allergies.dairy, 
        req.allergies.nuts, 
        req.allergies.peanut, 
        req.allergies.seafood, 
        req.allergies.egg, 
        req.allergies.soy, 
        req.allergies.celery
    ]
    
    ratios = [
        ("Sarapan (25%)", 0.25, req.breakfast_prefs), 
        ("Makan Siang (40%)", 0.40, req.lunch_prefs), 
        ("Makan Malam (35%)", 0.35, req.dinner_prefs)
    ]
    daily_plan_response = []
    
    used_foods = set() # Set untuk melacak makanan yang sudah direkomendasikan
    
    for meal_name, ratio, prefs in ratios:
        # Scale the macros for the specific meal
        meal_macros = [m * ratio for m in macro_array]
        
        # === APLIKASIKAN PREFERENSI SPESIFIK WAKTU MAKAN ===
        cats = prefs.food_category if prefs else []
        ings = prefs.main_ingredients if prefs else []
        
        # Jika AI mendeteksi waktu makan (misal "Makan Siang"), terapkan keyword hanya untuk sesi tersebut
        apply_ai_keywords = True
        if target_meal_from_ai and target_meal_from_ai.lower() not in meal_name.lower():
            apply_ai_keywords = False
            
        if apply_ai_keywords:
            ings = ings + extracted_keywords
            
        # Ambil nama bersih waktu makan (misal "Makan Siang") untuk DS logic
        clean_meal_name = meal_name.split(' (')[0]
        
        meal_specific_df = data_service.filter_foods_by_preferences(
            categories=cats, 
            ingredients=ings, 
            target_meal=clean_meal_name,
            dataset=food_df,
            negative_keywords=negative_keywords if apply_ai_keywords else []
        )
        
        # === OPSI 2: LOGIKA HEURISTIK PER WAKTU MAKAN ===
        if "Sarapan" in meal_name:
            # Sarapan: Makanan ringan/menengah (< 350 kkal per 100g)
            mask = meal_specific_df['calories_100g'] <= 350
            if not meal_specific_df[mask].empty:
                meal_specific_df = meal_specific_df[mask]
        elif "Makan Siang" in meal_name:
            # Makan Siang: Makanan berat (> 200 kkal per 100g)
            mask = meal_specific_df['calories_100g'] >= 200
            if not meal_specific_df[mask].empty:
                meal_specific_df = meal_specific_df[mask]
        elif "Makan Malam" in meal_name:
            # Makan Malam: Menengah (bebas, tapi hindari yang super berat > 500)
            mask = meal_specific_df['calories_100g'] <= 500
            if not meal_specific_df[mask].empty:
                meal_specific_df = meal_specific_df[mask]
        
        # Minta lebih banyak (top 15) agar kita bisa membuang yang duplikat
        meal_recs = ml_model.get_ai_recommendations(
            user_macros=meal_macros,
            user_allergies=allergy_array,
            filtered_food_df=meal_specific_df, # Gunakan DF yang sudah di-filter khusus waktu makan
            user_max=user_max,
            top_k=15
        )
        
        # Format the models & Enforce Variety
        formatted_recs = []
        for r in meal_recs:
            if r['food_name'] not in used_foods:
                formatted_recs.append(FoodRecommendation(
                    food_name=r['food_name'],
                    calories_100g=r['calories_100g'],
                    ideal_grams=r['ideal_grams'],
                    ideal_calories=r['ideal_calories'],
                    match_score=r['match_score']
                ))
                used_foods.add(r['food_name']) # Catat agar tidak muncul di waktu makan selanjutnya
                
            if len(formatted_recs) == 2: # Ambil 2 menu unik terbaik per waktu makan
                break
            
        daily_plan_response.append({
            "meal_name": meal_name,
            "target_calories": meal_macros[0],
            "recommendations": formatted_recs
        })
    
    if not daily_plan_response[0]["recommendations"]:
        return RecommendResponse(
            daily_plan=[],
            narrative_summary="Maaf, kami tidak menemukan makanan yang aman dengan preferensi dan alergi Anda."
        )
        
    # Ambil makanan terbaik dari Makan Siang untuk dijadikan bahan cerita Gen AI
    top_lunch_food = daily_plan_response[1]["recommendations"][0]
    
    # 4. Gen AI Narrative Generation
    narrative = genai_service.generate_narrative(
        user_text=req.user_text,
        recommended_food=top_lunch_food.food_name,
        portion=top_lunch_food.ideal_grams,
        target_cal=req.target_macros.calories # Mention total daily calories
    )
        
    return RecommendResponse(
        daily_plan=daily_plan_response,
        narrative_summary=narrative
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
