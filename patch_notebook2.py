import json
import codecs

with codecs.open('NutriMatch_Food_Recommender.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

new_code_8 = """# 7. Real Case Recommendation & Meal Planning
# Implementing Portion Sizing and Meal Time Distribution (Breakfast, Lunch, Dinner)

loaded_model = tf.keras.models.load_model(
    'nutrimatch_advanced_model.keras', 
    custom_objects={'InteractionLayer': InteractionLayer, 'asymmetric_allergy_loss': asymmetric_allergy_loss}
)

def get_meal_recommendations(meal_name, target_macros, user_allergy_list, top_k=2):
    # target_macros: [cal, prot, fat, carb]
    u_mac = np.array(target_macros, dtype=np.float32) / user_max
    u_all = np.array(user_allergy_list, dtype=np.float32)
    
    num_foods = len(food_subset)
    u_mac_batch = np.tile(u_mac, (num_foods, 1))
    u_all_batch = np.tile(u_all, (num_foods, 1))
    f_mac_batch = X_food_macro[:num_foods]
    f_all_batch = X_food_allergy[:num_foods]
    
    # Predict matching score using the AI model
    match_scores, allergy_risks = loaded_model.predict(
        [u_mac_batch, u_all_batch, f_mac_batch, f_all_batch], 
        batch_size=64, verbose=0
    )
    
    # Guardrail: Check for overlapping allergies
    has_allergy = np.any(np.logical_and(u_all_batch == 1, f_all_batch == 1), axis=1)
    
    results = []
    target_cal = target_macros[0]
    
    for i in range(num_foods):
        if not has_allergy[i]: # Guardrail blocks allergic foods
            cal_100g = food_subset.iloc[i]['calories_100g']
            if cal_100g > 0: 
                # Feature 1: Portion Sizing Calculation
                ideal_grams = (target_cal / cal_100g) * 100
                
                ideal_prot = (food_subset.iloc[i]['protein_100g'] / 100) * ideal_grams
                ideal_fat = (food_subset.iloc[i]['fat_100g'] / 100) * ideal_grams
                ideal_carb = (food_subset.iloc[i]['carbohydrate_100g'] / 100) * ideal_grams
                
                results.append({
                    'food_name': food_subset.iloc[i]['food_name'],
                    'ideal_grams': ideal_grams,
                    'ideal_prot': ideal_prot,
                    'ideal_fat': ideal_fat,
                    'ideal_carb': ideal_carb,
                    'match_score': match_scores[i][0]
                })
            
    # Sort by AI Match Score
    results = sorted(results, key=lambda x: x['match_score'], reverse=True)
    
    print(f"\\n🍽️ === {meal_name} (Target: {target_cal:.0f} kcal) ===")
    if not results:
        print("   [!] Tidak ada makanan yang aman dari alergi Anda di database.")
        return
        
    for i, res in enumerate(results[:top_k]):
        print(f"Opsi {i+1}: {res['food_name'].title()}")
        print(f"   -> Porsi Ideal: {res['ideal_grams']:.0f} gram")
        print(f"   -> Gizi didapat: {res['ideal_prot']:.1f}g Prot | {res['ideal_fat']:.1f}g Lemak | {res['ideal_carb']:.1f}g Karbo")
        print(f"   -> AI Match Score: {res['match_score']*100:.2f}%")

def generate_daily_meal_plan(daily_macros, user_allergy_list):
    print(f"=== 📅 JADWAL MAKAN HARIAN ===")
    print(f"Target Total: {daily_macros[0]} Kalori")
    print(f"Alergi [Gluten, Dairy, Nuts, Peanut, Seafood, Egg, Soy, Celery]: {user_allergy_list}")
    
    # Feature 2: Meal Time Distribution
    # Membagi persentase kalori harian
    ratios = {'Sarapan Pagi (25%)': 0.25, 'Makan Siang (40%)': 0.40, 'Makan Malam (35%)': 0.35}
    
    for meal_name, ratio in ratios.items():
        # Skalakan seluruh target gizi berdasarkan persentase jadwal makan
        meal_macros = [m * ratio for m in daily_macros]
        get_meal_recommendations(meal_name, meal_macros, user_allergy_list, top_k=2)

# ---- TEST CASE 1: User sehat tanpa alergi ----
print("\\n" + "="*50)
print("TEST CASE 1: User Biasa")
my_daily_macros_1 = [1800, 100, 50, 200]
my_allergies_1 = [0, 0, 0, 0, 0, 0, 0, 0] # Tidak ada alergi
generate_daily_meal_plan(my_daily_macros_1, my_allergies_1)

# ---- TEST CASE 2: User dengan pantangan ----
print("\\n" + "="*50)
print("TEST CASE 2: User Alergi Seafood & Telur")
my_daily_macros_2 = [2200, 120, 60, 250]
my_allergies_2 = [0, 0, 0, 0, 1, 1, 0, 0] # Alergi Seafood & Telur
generate_daily_meal_plan(my_daily_macros_2, my_allergies_2)
"""

# replace the last cell
notebook['cells'][-1]['source'] = new_code_8.splitlines(True)

with codecs.open('NutriMatch_Food_Recommender.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)
