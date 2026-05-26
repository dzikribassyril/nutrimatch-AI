import requests
import json
import time

url = "http://127.0.0.1:8000/api/recommend"

time.sleep(2)

payload = {
  "target_macros": {
    "calories": 2000,
    "protein_g": 120,
    "fat_g": 60,
    "carb_g": 250
  },
  "allergies": {
    "gluten": 0,
    "dairy": 0,
    "nuts": 0,
    "peanut": 0,
    "seafood": 0,
    "egg": 0,
    "soy": 0,
    "celery": 0
  },
  "breakfast_prefs": {
    "food_category": [],
    "main_ingredients": []
  },
  "lunch_prefs": {
    "food_category": [],
    "main_ingredients": []
  },
  "dinner_prefs": {
    "food_category": [],
    "main_ingredients": []
  },
  "user_text": "Saya ingin makan ayam di siang hari TAPI JANGAN digoreng"
}

headers = {
    "Content-Type": "application/json"
}

print("=== MENGUJI DENGAN NEGATIVE PREFERENCE ===")
print("User Text: Saya ingin makan ayam di siang hari TAPI JANGAN digoreng\n")

print("Mengirim request ke API...")
try:
    response = requests.post(url, json=payload, headers=headers)
    print(f"Status Code: {response.status_code}\n")
    if response.status_code == 200:
        data = response.json()
        print("=== NARRATIVE SUMMARY ===")
        print(data.get("narrative_summary", ""))
        print("\n=== DAILY PLAN ===")
        for meal in data.get("daily_plan", []):
            print(f"\n{meal['meal_name']} - Target Cal: {meal['target_calories']:.0f}")
            if not meal['recommendations']:
                print("  - TIDAK ADA REKOMENDASI (Filter terlalu ketat)")
            for rec in meal['recommendations']:
                print(f"  - {rec['food_name']} ({rec['ideal_grams']:.1f}g)")
    else:
        print("Error Response:")
        print(response.text)
except Exception as e:
    print(f"Error connecting to API: {e}")
