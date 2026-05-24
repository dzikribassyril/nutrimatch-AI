import os
from typing import List

# Import the new SDK
from google import genai

# Setup Gemini API Key
API_KEY = os.getenv("GEMINI_API_KEY", "") 

if API_KEY:
    client = genai.Client(api_key=API_KEY)
else:
    client = None
    print("Warning: GEMINI_API_KEY not found in environment. GenAI features will run in Mock Mode.")

def extract_food_preference(user_text: str) -> List[str]:
    """
    Uses Gemini to extract keywords (ingredients, category, taste) from natural language.
    Returns a list of keywords.
    """
    if not client or not user_text:
        return []
        
    prompt = f"""
    Kamu adalah asisten nutrisi. Ekstrak KATA KUNCI bahan makanan, kategori (misal: sup, mie, nasi), 
    atau rasa dari kalimat berikut.
    Kalimat: "{user_text}"
    
    Keluarkan hanya keywordnya saja, dipisahkan dengan koma (tanpa penjelasan tambahan).
    Jika tidak ada keyword makanan, kembalikan kosong.
    Contoh output: ayam, pedas, berkuah
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        text = response.text.strip()
        if text:
            return [kw.strip() for kw in text.split(',') if kw.strip()]
        return []
    except Exception as e:
        print(f"GenAI Extraction Error: {e}")
        return []

def generate_narrative(user_text: str, recommended_food: str, portion: float, target_cal: float) -> str:
    """
    Uses Gemini to craft a friendly response explaining the recommendation.
    """
    if not client:
        # Mock Response
        return f"Kami merekomendasikan {recommended_food} ({portion:.0f} gram) untuk memenuhi target {target_cal:.0f} kalori Anda."

    prompt = f"""
    Kamu adalah ahli gizi ramah bernama 'NutriAI'. 
    Pengguna meminta: "{user_text if user_text else 'Rekomendasi makanan sehat'}"
    Berdasarkan AI kami, makanan terbaik dan paling aman adalah "{recommended_food}" dengan porsi "{portion:.0f} gram".
    Target kalori yang dipenuhi adalah "{target_cal:.0f} kalori".
    
    Tulis 2 kalimat ramah yang memberi tahu pengguna untuk makan makanan tersebut dengan porsi yang disebut 
    agar target dietnya tercapai dan seleranya terpenuhi. Gunakan bahasa Indonesia kasual yang asik.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        print(f"GenAI Narrative Error: {e}")
        return f"Cobalah {recommended_food} sebanyak {portion:.0f} gram ya!"
