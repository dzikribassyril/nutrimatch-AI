import requests
import json
import time

url = "http://127.0.0.1:8002/api/recommend"

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
  "user_text": "Saya ingin makan ayam di siang hari TAPI JANGAN digoreng",
  "start_date": "2026-05-28",
  "days": 7,
  "variety_penalty": 0.15
}

headers = {
    "Content-Type": "application/json"
}

print("=== MENGUJI DENGAN NEGATIVE PREFERENCE ===")
print("User Text: Saya ingin makan ayam di siang hari TAPI JANGAN digoreng\n")

print("Mengirim request ke API...")
max_retries = 30
for attempt in range(1, max_retries + 1):
    try:
        response = requests.post(url, json=payload, headers=headers)
        print(f"Status Code: {response.status_code}\n")
        if response.status_code == 200:
            data = response.json()
            print("=== NARRATIVE SUMMARY ===")
            print(data.get("narrative_summary", ""))
            print("\n=== DAILY PLAN ===")
            current_day = None
            for meal in data.get("daily_plan", []):
              name_parts = meal.get("meal_name", "").split(" - ")
              if len(name_parts) >= 3:
                day_title = " - ".join(name_parts[:2])
                meal_title = name_parts[2]
              else:
                day_title = "Hari"
                meal_title = meal.get("meal_name", "")

              if day_title != current_day:
                current_day = day_title
                print(f"\n{day_title}")

              print(f"  {meal_title} - Target Cal: {meal['target_calories']:.0f}")
              if not meal['recommendations']:
                print("    - TIDAK ADA REKOMENDASI (Filter terlalu ketat)")
              for rec in meal['recommendations']:
                p_str = f"P:{rec['ideal_protein']:.1f}g L:{rec['ideal_fat']:.1f}g K:{rec['ideal_carb']:.1f}g" if rec.get('ideal_protein') is not None else ""
                print(f"    - {rec['food_name']} ({rec['ideal_grams']:.1f}g | {rec['ideal_calories']:.0f} kcal | {p_str})")
            break
        else:
            print("Error Response:")
            print(response.text)
            break
    except requests.exceptions.ConnectionError:
        if attempt < max_retries:
            print(f"  [Attempt {attempt}/{max_retries}] Server not ready, retrying in 2 seconds...")
            time.sleep(2)
        else:
            print("Error: Could not connect to API after multiple retries.")
    except Exception as e:
        print(f"Error connecting to API: {e}")
        break
