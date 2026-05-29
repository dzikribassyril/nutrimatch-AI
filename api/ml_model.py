import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

import tensorflow as tf
import numpy as np
import pandas as pd
import random

class InteractionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(InteractionLayer, self).__init__(**kwargs)
    def call(self, inputs):
        user_embed, food_embed = inputs
        return tf.multiply(user_embed, food_embed)

def asymmetric_allergy_loss(y_true, y_pred):
    y_true = tf.cast(tf.reshape(y_true, [-1, 1]), tf.float32)
    fn_weight = 10.0
    fp_weight = 1.0
    weights = y_true * fn_weight + (1 - y_true) * fp_weight
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(weights[:, 0] * bce)

FOOD_MACRO_COLS = ['calories_100g', 'protein_100g', 'fat_100g', 'carbohydrate_100g']
ALLERGY_COLS_FOOD = [
    'contains_gluten', 'contains_dairy', 'contains_nuts', 'contains_peanut',
    'contains_seafood', 'contains_egg', 'contains_soy', 'contains_celery'
]

MEAL_RATIOS = {
    'breakfast': ('Sarapan (25%)', 0.25),
    'lunch': ('Makan Siang (40%)', 0.40),
    'dinner': ('Makan Malam (35%)', 0.35),
}

DAY_NAMES = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'nutrimatch_model.keras')

try:
    loaded_model = tf.keras.models.load_model(
        MODEL_PATH, 
        custom_objects={'InteractionLayer': InteractionLayer, 'asymmetric_allergy_loss': asymmetric_allergy_loss}
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    loaded_model = None

def is_valid_for_role(row: pd.Series, role: str) -> bool:
    """
    Evaluasi canggih berbasis kategori masakan dan kata spesifik.
    """
    fname_lower = str(row.get('food_name', '')).lower()
    cat_lower = str(row.get('food_category', '')).lower()
    cook_cat_lower = str(row.get('cooking_category', '')).lower()

    # 1. Global Rejections (Cegah hal yang tidak bisa dimakan sebagai makanan pokok)
    INVALID_FOOD_CATEGORIES = ['minyak', 'lemak', 'bumbu_sambal', 'bumbu', 'snack_dessert', 'minuman', 'kue', 'jajanan']
    INVALID_NAMES = ['tepung', 'mentah', 'kering', 'bubuk', 'sirup', 'kaldu', 'gula', 'garam', 'kerupuk', 'keripik', 'beras', 'mentega', 'margarin', 'dideh', 'ampas', 'abon', 'minyak', 'atom']
    
    if any(invalid in cat_lower for invalid in INVALID_FOOD_CATEGORIES): return False
    if any(invalid in fname_lower for invalid in INVALID_NAMES): return False
    
    # Blokir makanan mentah kecuali untuk sayuran/buah (lalapan)
    if cook_cat_lower == 'mentah_segar' and role in ['Complete', 'Karbo', 'Lauk']: return False

    # 2. Complete / Makanan Tunggal (One-dish meal)
    VALID_COMPLETE_NAMES = ['nasi goreng', 'mie goreng', 'soto', 'sup', 'bubur', 'burger', 'pizza', 'nasi uduk', 'nasi kuning', 'nasi campur', 'nasi padang', 'nasi gemuk', 'mie ayam', 'bakso', 'sate', 'spaghetti', 'macaroni', 'lontong sayur', 'ketupat sayur', 'martabak']
    is_complete_by_name = any(c in fname_lower for c in VALID_COMPLETE_NAMES)
    is_complete_by_cat = ('berkuah' in cat_lower)

    if role == 'Complete': 
        return is_complete_by_cat or is_complete_by_name
        
    # Jika sudah diklasifikasikan sebagai hidangan lengkap, jangan jadikan dia lauk atau karbo lagi (mencegah "Soto" direkomendasi sebagai lauk kombo)
    if is_complete_by_cat or is_complete_by_name: 
        return False

    # Split kata untuk menghindari "daun singkong" menjadi "singkong" (karbo)
    words = fname_lower.replace('-', ' ').split()

    # 3. Kategori Komponen
    if role == 'Karbo':
        if 'karbohidrat_pokok' in cat_lower: return True
        return any(k in words for k in ['nasi', 'mie', 'roti', 'pasta', 'kentang', 'umbi', 'singkong', 'ubi', 'sereal', 'oat', 'gandum']) and cat_lower not in ['sayuran', 'buah', 'lauk_hewani', 'lauk_nabati']

    if role == 'Lauk':
        if cat_lower in ['lauk_hewani', 'lauk_nabati']: return True
        return any(l in words for l in ['daging', 'ayam', 'ikan', 'seafood', 'tahu', 'tempe', 'telur', 'kacang', 'sapi', 'kambing', 'udang', 'cumi']) and cat_lower not in ['sayuran', 'buah', 'karbohidrat_pokok']

    if role == 'Sayur':
        if cat_lower in ['sayuran', 'buah', 'tumis']: return True
        return any(s in words for s in ['sayur', 'bayam', 'kangkung', 'brokoli', 'wortel', 'tomat', 'kol', 'sawi', 'selada', 'jamur', 'salad', 'apel', 'pisang', 'jeruk', 'pepaya']) and cat_lower not in ['lauk_hewani', 'lauk_nabati', 'karbohidrat_pokok']

    return False


def _normalize_food_macros(filtered_food_df: pd.DataFrame) -> np.ndarray:
    f_mac = filtered_food_df.reindex(columns=FOOD_MACRO_COLS, fill_value=0).fillna(0).values.astype(np.float32)
    food_max = np.nanmax(f_mac, axis=0)
    food_max = np.where(food_max <= 0, 1.0, food_max)
    return f_mac / food_max

def _build_food_allergy_matrix(filtered_food_df: pd.DataFrame) -> np.ndarray:
    return filtered_food_df.reindex(columns=ALLERGY_COLS_FOOD, fill_value=0).fillna(0).astype(np.float32).values

def _build_meal_suitable_mask(filtered_food_df: pd.DataFrame, meal_type: str) -> np.ndarray:
    meal_col_map = {'breakfast': 'suitable_breakfast', 'lunch': 'suitable_lunch', 'dinner': 'suitable_dinner'}
    suitable_col = meal_col_map.get(meal_type)
    if not suitable_col or suitable_col not in filtered_food_df.columns:
        return np.ones(len(filtered_food_df), dtype=bool)

    col = filtered_food_df[suitable_col].fillna(1)
    if col.dtype != bool:
        col = col.astype(int)
    return col.astype(bool).values

def get_combo_meal_recommendations(
    meal_type: str, target_macros: list, user_allergy_list: list, 
    filtered_food_df: pd.DataFrame, user_max: np.ndarray, 
    used_foods: set, variety_penalty: float = 0.20
) -> Dict[str, Dict]:
    
    if loaded_model is None or filtered_food_df.empty:
        return {}

    num_foods = len(filtered_food_df)
    u_all = np.array(user_allergy_list, dtype=np.float32)
    u_all_batch = np.tile(u_all, (num_foods, 1))

    f_mac = _normalize_food_macros(filtered_food_df)
    f_all = _build_food_allergy_matrix(filtered_food_df)
    suitable_mask = _build_meal_suitable_mask(filtered_food_df, meal_type)
    has_allergy = np.any(np.logical_and(u_all_batch == 1, f_all == 1), axis=1)

    def predict_scores(t_macro: list) -> np.ndarray:
        u_mac = np.array(t_macro, dtype=np.float32) / user_max
        u_mac_batch = np.tile(u_mac, (num_foods, 1))
        scores, _ = loaded_model.predict([u_mac_batch, u_all_batch, f_mac, f_all], batch_size=256, verbose=0)
        return scores.flatten()

    target_k = [target_macros[0] * 0.45, target_macros[1] * 0.10, target_macros[2] * 0.10, target_macros[3] * 0.60]
    target_l = [target_macros[0] * 0.40, target_macros[1] * 0.75, target_macros[2] * 0.70, target_macros[3] * 0.10]
    target_s = [target_macros[0] * 0.15, target_macros[1] * 0.15, target_macros[2] * 0.20, target_macros[3] * 0.30]
    
    scores_full = predict_scores(target_macros)
    scores_k = predict_scores(target_k)
    scores_l = predict_scores(target_l)
    scores_s = predict_scores(target_s)

    full_cands, k_cands, l_cands, s_cands = [], [], [], []

    for i in range(num_foods):
        if has_allergy[i] or not suitable_mask[i]:
            continue

        row = filtered_food_df.iloc[i]
        fname = row.get('food_name', '')
        cal_100g = float(row.get('calories_100g', 0))
        if cal_100g <= 0:
            continue

        penalty = variety_penalty if fname in used_foods else 0

        if is_valid_for_role(row, 'Complete'):
            full_cands.append((i, fname, scores_full[i] - penalty, row))
        if is_valid_for_role(row, 'Karbo'):
            k_cands.append((i, fname, scores_k[i] - penalty, row))
        if is_valid_for_role(row, 'Lauk'):
            l_cands.append((i, fname, scores_l[i] - penalty, row))
        if is_valid_for_role(row, 'Sayur'):
            s_cands.append((i, fname, scores_s[i] - penalty, row))

    full_cands.sort(key=lambda x: x[2], reverse=True)
    k_cands.sort(key=lambda x: x[2], reverse=True)
    l_cands.sort(key=lambda x: x[2], reverse=True)
    s_cands.sort(key=lambda x: x[2], reverse=True)

    use_combo = True
    best_full_score = full_cands[0][2] if full_cands else -1
    avg_combo_score = -1

    if k_cands and l_cands and s_cands:
        avg_combo_score = (k_cands[0][2] + l_cands[0][2] + s_cands[0][2]) / 3

    if full_cands and (best_full_score > avg_combo_score * 0.95 or not k_cands or not l_cands):
        use_combo = False

    result = {}
    if not use_combo and full_cands:
        best = full_cands[0]
        grams = (target_macros[0] / float(best[3].get('calories_100g', 1))) * 100
        grams = min(max(grams, 200), 600)
        
        result['Makanan Utama'] = {
            'food_name': best[1],
            'calories_100g': float(best[3].get('calories_100g', 0)),
            'grams': float(grams),
            'cal': float((best[3].get('calories_100g', 0) / 100) * grams),
            'score': float(best[2]),
        }
    else:
        for role, target_m, cands in zip(['Karbo', 'Lauk', 'Sayur'], [target_k, target_l, target_s], [k_cands, l_cands, s_cands]):
            if not cands: continue
            best = cands[0]
            grams = (target_m[0] / float(best[3].get('calories_100g', 1))) * 100
            grams = min(max(grams, 50), 250)

            result[role] = {
                'food_name': best[1],
                'calories_100g': float(best[3].get('calories_100g', 0)),
                'grams': float(grams),
                'cal': float((best[3].get('calories_100g', 0) / 100) * grams),
                'score': float(best[2]),
            }

    return result

def generate_7day_combo_plan(
    daily_macros: list, user_allergy_list: list, meal_datasets: Optional[Dict[str, pd.DataFrame]],
    user_max: np.ndarray, variety_penalty: float = 0.20, start_date: Optional[str] = None, days: int = 7,
) -> List[Dict]:
    if loaded_model is None or days <= 0: return []

    base_date = datetime.strptime(start_date, '%Y-%m-%d') if start_date else datetime.today()
    used_foods, plan = set(), []

    for day_idx in range(days):
        current_date = base_date + timedelta(days=day_idx)
        day_label = DAY_NAMES[current_date.weekday()]
        day_used = set()

        for meal_key, (meal_label, ratio) in MEAL_RATIOS.items():
            meal_macros = [m * ratio for m in daily_macros]
            meal_df = meal_datasets.get(meal_key) if meal_datasets else None
            if meal_df is None: continue

            combo = get_combo_meal_recommendations(
                meal_type=meal_key, target_macros=meal_macros, user_allergy_list=user_allergy_list,
                filtered_food_df=meal_df, user_max=user_max, used_foods=used_foods | day_used,
                variety_penalty=variety_penalty,
            )

            recommendations = []
            for role in ['Makanan Utama', 'Karbo', 'Lauk', 'Sayur']:
                if role not in combo: continue
                item = combo[role]
                recommendations.append({
                    'food_name': f"[{role}] {item['food_name']}",
                    'calories_100g': item['calories_100g'],
                    'ideal_grams': item['grams'],
                    'ideal_calories': item['cal'],
                    'match_score': item['score'],
                })
                day_used.add(item['food_name'])

            plan.append({
                'meal_name': f"Hari {day_idx + 1} - {day_label} - {meal_label}",
                'target_calories': float(meal_macros[0]),
                'recommendations': recommendations,
            })

        used_foods |= day_used

    return plan