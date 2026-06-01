"""Quick inline test of the updated API pipeline (no HTTP server needed)."""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

import data_service
import ml_model

food_df = data_service.get_food_data()
user_max = data_service.get_user_max()

print(f"Food rows: {len(food_df)}")
print(f"User max: {user_max}")
print()

# --- Build meal datasets (no AI filtering, just meal-time suitability) ---
meal_datasets = {}
for meal_key in ['breakfast', 'lunch', 'dinner']:
    clean_name = {'breakfast': 'Sarapan', 'lunch': 'Makan Siang', 'dinner': 'Makan Malam'}[meal_key]
    meal_datasets[meal_key] = data_service.filter_foods_by_preferences(
        categories=[], ingredients=[], target_meal=clean_name, dataset=food_df,
    )
    print(f"  {clean_name}: {len(meal_datasets[meal_key])} rows")

print()

# --- Test Case 1: User biasa tanpa alergi ---
print("=" * 60)
print("TEST CASE 1: User Biasa (Tanpa Alergi) — 1800 kcal")
print("=" * 60)

plan = ml_model.generate_7day_combo_plan(
    daily_macros=[1800, 100, 50, 200],
    user_allergy_list=[0, 0, 0, 0, 0, 0, 0, 0],
    meal_datasets=meal_datasets,
    user_max=user_max,
    variety_penalty=0.15,
    days=3,  # 3 hari saja untuk cepat
)

for meal in plan:
    print(f"\n{meal['meal_name']} (Target: {meal['target_calories']:.0f} kcal)")
    if not meal['recommendations']:
        print("  ⚠️ Tidak ada rekomendasi")
    for rec in meal['recommendations']:
        name = rec['food_name']
        g = rec['ideal_grams']
        cal = rec['ideal_calories']
        p = rec.get('ideal_protein', 0)
        f = rec.get('ideal_fat', 0)
        c = rec.get('ideal_carb', 0)
        s = rec['match_score']
        print(f"  {name:45s}  {g:5.0f}g | {cal:5.0f} kcal | P:{p:5.1f}g L:{f:5.1f}g K:{c:5.1f}g | Score: {s*100:.1f}%")

print()
print("=" * 60)
print("TEST CASE 2: User Alergi Seafood & Telur — 2200 kcal")
print("=" * 60)

plan2 = ml_model.generate_7day_combo_plan(
    daily_macros=[2200, 120, 60, 250],
    user_allergy_list=[0, 0, 0, 0, 1, 1, 0, 0],
    meal_datasets=meal_datasets,
    user_max=user_max,
    variety_penalty=0.15,
    days=3,
)

for meal in plan2:
    print(f"\n{meal['meal_name']} (Target: {meal['target_calories']:.0f} kcal)")
    if not meal['recommendations']:
        print("  ⚠️ Tidak ada rekomendasi")
    for rec in meal['recommendations']:
        name = rec['food_name']
        g = rec['ideal_grams']
        cal = rec['ideal_calories']
        p = rec.get('ideal_protein', 0)
        f = rec.get('ideal_fat', 0)
        c = rec.get('ideal_carb', 0)
        s = rec['match_score']
        print(f"  {name:45s}  {g:5.0f}g | {cal:5.0f} kcal | P:{p:5.1f}g L:{f:5.1f}g K:{c:5.1f}g | Score: {s*100:.1f}%")

print("\n== Test selesai! ==")
