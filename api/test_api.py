from fastapi.testclient import TestClient
from main import app
import json

# Menggunakan TestClient bawaan FastAPI agar tidak perlu menyalakan uvicorn server
client = TestClient(app)

def test_manual_input():
    print("=== MENGUJI API DENGAN INPUT MANUAL (TANPA GEN AI) ===")
    
    # Payload (Body Request) simulasi dari Web Frontend
    payload = {
        "target_macros": {
            "calories": 2000.0,
            "protein_g": 150.0,
            "fat_g": 60.0,
            "carb_g": 200.0
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
            "food_category": ["roti", "susu"],
            "main_ingredients": []
        },
        "lunch_prefs": {
            "food_category": ["gorengan"],
            "main_ingredients": ["ayam", "sapi"]
        },
        "dinner_prefs": None, # Opsional, bisa dikosongkan!
        "user_text": None
    }
    
    print(f"Mengirim Data:\n{json.dumps(payload, indent=2)}\n")
    
    # Melakukan POST request ke rute /api/recommend
    response = client.post("/api/recommend", json=payload)
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Response JSON (Berhasil):")
        print(json.dumps(response.json(), indent=2))
    else:
        print("Response Error:")
        print(response.text)

if __name__ == "__main__":
    test_manual_input()
