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

import json
from schemas import ExtractedPreferences

def extract_food_preference(user_text: str) -> ExtractedPreferences:
    """Uses GenAI to extract keywords, negative keywords, and target meal times."""
    
    prompt = f"""
    Ekstrak informasi dari teks pengguna mengenai preferensi makanan.
    Fokus pada:
    1. keywords: jenis makanan utama, bahan dasar, atau kategori yang diinginkan (misal: "ayam", "berkuah", "pedas").
    2. negative_keywords: jenis makanan atau bahan yang TIDAK diinginkan atau dilarang (misal kata kunci setelah kata "jangan", "tanpa", "bukan", contoh: "goreng", "santan"). Selalu gunakan kata dasar (misal: "digoreng" menjadi "goreng", "direbus" menjadi "rebus").
    3. target_meal: Jika ada penyebutan waktu makan yang spesifik, pilih dari: "Sarapan", "Makan Siang", "Makan Malam". Jika tidak ada, biarkan null.

    Kembalikan HANYA format JSON berikut, tanpa markdown, tanpa teks lain:
    {{
        "keywords": ["kata1", "kata2"],
        "negative_keywords": ["kata3", "kata4"],
        "target_meal": "Makan Siang"
    }}

    Teks pengguna: "{user_text}"
    """
    
    if not client or not user_text:
        return ExtractedPreferences(keywords=[], negative_keywords=[], target_meal=None)

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        text = response.text.strip()
        # Bersihkan format markdown jika Gemini masih menambahkannya
        if text.startswith("```json"):
            text = text.replace("```json", "").replace("```", "").strip()
        elif text.startswith("```"):
            text = text.replace("```", "").strip()
            
        data = json.loads(text)
        return ExtractedPreferences(
            keywords=data.get("keywords", []),
            negative_keywords=data.get("negative_keywords", []),
            target_meal=data.get("target_meal")
        )
    except Exception as e:
        print(f"GenAI Extraction Error: {e}")
        return ExtractedPreferences(keywords=[], negative_keywords=[], target_meal=None)

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
