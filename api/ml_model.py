import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import tensorflow as tf
import numpy as np
import pandas as pd

# ============================================================
# Custom Layer & Loss (harus match dengan model .keras)
# ============================================================
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

# ============================================================
# Constants
# ============================================================
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

TRADITIONAL_STAPLE_GROUPS = frozenset(['rice_noodle_staple', 'savory_staple'])
SWEET_STAPLE_GROUPS = frozenset(['sweet_snack', 'bread_staple'])
TRADITIONAL_PROTEIN_GROUPS = frozenset(['traditional_protein'])
TRADITIONAL_VEGETABLE_GROUPS = frozenset(['cooked_or_ready_vegetable'])
DAIRY_PAIRING_GROUPS = frozenset(['dairy_pairing'])
TOP_PAIRING_CANDIDATES = 24

# ============================================================
# Load Model
# ============================================================
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


# ============================================================
# IMPROVED is_valid_for_role — Rule-Based Guardrails v2
# ============================================================
# Menggunakan kolom dataset v2:
#   - recommendation_item_type  (menu, staple, protein_dish, plant_protein,
#                                vegetable, fruit, snack, drink, local_food_or_menu)
#   - food_category             (karbohidrat_pokok, lauk_hewani, lauk_nabati,
#                                sayuran, buah, gorengan, berkuah, lainnya, ...)
#   - Nutrisi makro             (calories_100g, protein_100g, fat_100g, carbohydrate_100g)
# ============================================================

# ── Blacklists ───────────────────────────────────────────────
# Kategori makanan yang TIDAK boleh masuk komponen makan utama
_GLOBAL_REJECT_CATEGORIES = frozenset([
    'minyak', 'lemak', 'bumbu_sambal', 'bumbu', 'minuman',
])

# Item type yang TIDAK boleh jadi komponen makan utama (Karbo/Lauk/Sayur)
_REJECT_ITEM_TYPES_FOR_COMPONENTS = frozenset(['snack', 'drink'])

# Kata-kata dalam nama yang menandakan bahan mentah / bumbu / bukan makanan siap santap
_GLOBAL_REJECT_NAMES = frozenset([
    'mentega', 'margarin', 'minyak', 'sirup', 'kaldu', 'gula pasir',
    'garam', 'dideh', 'ampas', 'teh hijau', 'teh melati', 'daun kering',
    'jagung titi', 'katul jagung', 'enting-enting', 'risoles',
    'bihun goreng instan', 'getuk', 'tapai', 'lupis', 'ongol-ongol',
    'makaroni',
])

_RAW_STAPLE_INGREDIENTS = frozenset(['jagung'])
_RAW_STAPLE_MARKERS = frozenset(['pipil', 'giling', 'kering'])
_READY_TO_EAT_MARKERS = frozenset([
    'rebus', 'masak', 'matang', 'goreng', 'bakar', 'kukus', 'tumis',
    'panggang', 'nasi', 'bubur', 'lontong', 'ketupat',
])

# Kata dalam nama yang menandakan makanan BUKAN lauk/sayur yang tepat
_REJECT_LAUK_NAMES = frozenset([
    'noga', 'enting', 'dodol', 'bagea', 'ledre', 'kue', 'sale',
    'keripik', 'kerupuk', 'rempeyek', 'sukro', 'goyang', 'atom',
    'sangan', 'goreng kacang',
])

# Kata dalam nama yang menandakan makanan BUKAN sayur yang tepat
_REJECT_SAYUR_NAMES = frozenset([
    'noga', 'enting', 'dodol', 'bagea', 'ledre', 'kue', 'sale',
    'keripik', 'kerupuk', 'rempeyek', 'kelapa hutan', 'kelapa tua',
    'pipil', 'giling', 'kering', 'katul', 'biji nangka',
])

# Kata di nama yang menandakan makanan olahan gorengan/snack bukan karbohidrat pokok
_REJECT_KARBO_NAMES = frozenset([
    'keripik', 'kerupuk', 'ceriping', 'serimping', 'kripik',
    'rempeyek', 'kecimpring', 'lanting',
])

# ── Hidangan lengkap (Complete / One-dish meal) ──────────────
_COMPLETE_NAMES = [
    'nasi goreng', 'mie goreng', 'soto', 'sup ', 'bubur', 'burger', 'pizza',
    'nasi uduk', 'nasi kuning', 'nasi campur', 'nasi padang', 'nasi gemuk',
    'nasi gurih', 'mie ayam', 'bakso', 'sate', 'spaghetti', 'macaroni',
    'lontong sayur', 'ketupat sayur', 'martabak', 'gado-gado', 'gado gado',
    'pecel', 'rawon', 'rendang', 'opor', 'gulai', 'semur',
    'cap cay', 'capcay', 'tongseng', 'karedok',
]

# ── Keywords untuk identifikasi role via nama ────────────────
_KARBO_KEYWORDS = frozenset([
    'nasi', 'mie', 'roti', 'pasta', 'kentang', 'umbi', 'singkong', 'ubi',
    'sereal', 'oat', 'gandum', 'sagu', 'bihun', 'lontong', 'ketupat', 
    'ketan', 'talas', 'oyek',
])

_LAUK_KEYWORDS = frozenset([
    'daging', 'ayam', 'ikan', 'tahu', 'tempe', 'telur', 'sapi',
    'kambing', 'udang', 'cumi', 'bebek', 'kepiting', 'kerang',
    'bandeng', 'lele', 'tongkol', 'tuna', 'salmon', 'patin',
    'dendeng', 'rendang', 'semur', 'sate', 'opor',
])

_SAYUR_KEYWORDS = frozenset([
    'sayur', 'bayam', 'kangkung', 'brokoli', 'wortel', 'tomat', 'kol',
    'sawi', 'selada', 'jamur', 'salad', 'terong', 'labu', 'pare',
    'timun', 'mentimun', 'kacang panjang', 'buncis', 'tauge', 'toge',
    'daun', 'rebung', 'jengkol', 'petai',
])

_PRIMARY_RICE_NAMES = frozenset([
    'nasi',
    'nasi putih',
    'beras giling masak nasi',
])

_SECONDARY_RICE_NAMES = frozenset([
    'nasi beras merah',
])

_COMMON_READY_STAPLE_PHRASES = frozenset([
    'jagung rebus',
    'jagung muda rebus',
    'jagung kuning pipil rebus',
    'singkong kukus',
    'singkong goreng',
    'ubi jalar rebus',
    'ubi jalar kuning kukus',
    'ubi jalar tinta kemayung kukus',
    'talas kukus',
    'talas bogor kukus',
    'ketela pohonsingkong kukus',
    'ketupat ketan',
    'bihun goreng',
])


def _has_raw_staple_name(food_name: str) -> bool:
    name = str(food_name).lower().strip()
    if any(marker in name for marker in _READY_TO_EAT_MARKERS):
        return False
    return (
        any(ingredient in name for ingredient in _RAW_STAPLE_INGREDIENTS)
        and any(marker in name for marker in _RAW_STAPLE_MARKERS)
    )


def _karbo_priority_adjustment(row: pd.Series) -> float:
    fname_lower = str(row.get('food_name', '')).lower().strip()
    pairing_group = str(row.get('pairing_group', '')).lower().strip()

    if fname_lower in _PRIMARY_RICE_NAMES:
        return 0.30
    if fname_lower in _SECONDARY_RICE_NAMES:
        return 0.20
    if any(phrase in fname_lower for phrase in _COMMON_READY_STAPLE_PHRASES):
        return 0.25
    if pairing_group == 'rice_noodle_staple':
        return 0.10
    return 0.0


def _is_rice_family(row: pd.Series) -> bool:
    fname_lower = str(row.get('food_name', '')).lower().strip()
    main_ingredient = str(row.get('main_ingredient', '')).lower().strip()
    return (
        fname_lower in _PRIMARY_RICE_NAMES
        or fname_lower in _SECONDARY_RICE_NAMES
        or fname_lower.startswith('nasi ')
        or main_ingredient == 'beras'
    )


def _karbo_variety_penalty(row: pd.Series, used_foods: set) -> float:
    if not _is_rice_family(row):
        return 0.0

    used_lower = {str(food).lower().strip() for food in used_foods}
    rice_already_used = any(
        food in _PRIMARY_RICE_NAMES
        or food in _SECONDARY_RICE_NAMES
        or food.startswith('nasi ')
        or 'beras giling masak nasi' in food
        for food in used_lower
    )
    return 0.35 if rice_already_used else 0.0


def is_valid_for_role(row: pd.Series, role: str) -> bool:
    """
    Evaluasi apakah makanan valid untuk suatu role dalam Combo Meal.
    Menggunakan gabungan:
      1. recommendation_item_type (dari dataset v2)
      2. food_category
      3. Validasi makro nutrisi
      4. Keyword nama makanan
    """
    fname_lower = str(row.get('food_name', '')).lower().strip()
    cat_lower = str(row.get('food_category', '')).lower().strip()
    cook_cat = str(row.get('cooking_category', '')).lower().strip()
    item_type = str(row.get('recommendation_item_type', '')).lower().strip()

    cal = float(row.get('calories_100g', 0))
    prot = float(row.get('protein_100g', 0))
    fat = float(row.get('fat_100g', 0))
    carb = float(row.get('carbohydrate_100g', 0))

    # Pisah kata untuk word-level matching (hindari substring partial match)
    words = fname_lower.replace('-', ' ').split()
    is_recommendable = str(row.get('is_recommendable_food', 'true')).lower().strip()
    pairing_group = str(row.get('pairing_group', '')).lower().strip()
    pairing_role = str(row.get('pairing_role', '')).lower().strip()

    if is_recommendable in ('false', '0', '0.0'):
        return False

    # ── 1. Global Rejections ─────────────────────────────────
    # Reject kategori yang bukan makanan siap santap
    if cat_lower in _GLOBAL_REJECT_CATEGORIES:
        return False

    # Reject nama-nama bahan mentah/bumbu
    if any(reject in fname_lower for reject in _GLOBAL_REJECT_NAMES):
        return False

    # Reject bahan karbo mentah yang salah label sebagai staple/menu.
    if _has_raw_staple_name(fname_lower):
        return False

    # Zero-calorie atau negatif
    if cal <= 0:
        return False

    # Reject makanan mentah kecuali untuk sayur/pelengkap
    if cook_cat == 'mentah_segar' and role in ('Complete', 'Karbo', 'Lauk'):
        return False

    # Dataset v3 adds explicit meal-pairing metadata. Prefer it when available
    # so categories such as bread/snack/dairy cannot leak into main-meal combos.
    if pairing_role:
        if role == 'Complete':
            return pairing_role == 'complete' and pairing_group == 'complete_menu' and cal >= 80 and prot >= 5 and carb >= 10
        if role == 'Karbo':
            return pairing_role == 'staple' and pairing_group in TRADITIONAL_STAPLE_GROUPS and carb > fat * 1.2
        if role == 'Sweet':
            return (
                (pairing_role == 'sweet_snack' and pairing_group == 'sweet_snack')
                or (pairing_role == 'staple' and pairing_group == 'bread_staple')
            ) and cal > 0
        if role == 'Dairy':
            return pairing_role == 'dairy' and pairing_group in DAIRY_PAIRING_GROUPS and cal > 0
        if role == 'Lauk':
            return pairing_role == 'protein' and pairing_group in TRADITIONAL_PROTEIN_GROUPS and prot >= 8.0
        if role == 'Sayur':
            return pairing_role == 'vegetable' and pairing_group in TRADITIONAL_VEGETABLE_GROUPS and cal <= 300.0 and fat <= 25.0

    # ── 2. Complete (Hidangan Lengkap / One-dish meal) ───────
    is_complete_by_name = any(c in fname_lower for c in _COMPLETE_NAMES)
    is_complete_by_cat = ('berkuah' in cat_lower)

    if role == 'Complete':
        if not (is_complete_by_cat or is_complete_by_name):
            return False
        # Complete meal harus punya nutrisi yang cukup beragam
        return cal >= 80 and prot >= 5 and carb >= 10

    # Jika makanan ini adalah hidangan lengkap, jangan jadikan komponen
    if is_complete_by_cat or is_complete_by_name:
        return False

    # ── 3. Reject snack & drink untuk komponen utama ─────────
    if item_type in _REJECT_ITEM_TYPES_FOR_COMPONENTS:
        return False

    # ── 4. Role-specific validation ──────────────────────────

    # ───── KARBO ─────────────────────────────────────────────
    if role == 'Karbo':
        # Kategori snack_dessert, kue, jajanan → bukan karbohidrat pokok
        if cat_lower in ('snack_dessert', 'kue', 'jajanan'):
            return False

        # Reject olahan gorengan/keripik dari karbo
        if any(reject in fname_lower for reject in _REJECT_KARBO_NAMES):
            return False

        # Berdasarkan item_type
        if item_type == 'staple':
            return carb > fat * 1.5  # Karbo harus signifikan lebih besar dari lemak

        # Berdasarkan food_category
        if 'karbohidrat_pokok' in cat_lower:
            return carb > fat * 1.5

        # Berdasarkan keyword nama
        if any(k in words for k in _KARBO_KEYWORDS):
            # Pastikan bukan sayuran/buah/lauk dan karbo dominan
            if cat_lower in ('sayuran', 'buah', 'lauk_hewani', 'lauk_nabati'):
                return False
            return carb > fat * 1.5

        return False

    # ───── LAUK (Protein Source) ─────────────────────────────
    if role == 'Lauk':
        # Reject permen/camilan yang berlabel lauk_nabati
        if any(reject in fname_lower for reject in _REJECT_LAUK_NAMES):
            return False

        # Minimal protein requirement untuk lauk
        MIN_PROTEIN_LAUK = 8.0  # gram per 100g

        # Fat-dominance check: jika lemak sangat dominan (>2x protein),
        # ini bukan lauk yang baik (kemungkinan kacang goreng/snack)
        if fat > prot * 2.5 and fat > 30:
            return False

        # Berdasarkan item_type
        if item_type in ('protein_dish', 'plant_protein'):
            return prot >= MIN_PROTEIN_LAUK and prot > carb * 0.3

        # Berdasarkan food_category
        if cat_lower in ('lauk_hewani', 'lauk_nabati'):
            return prot >= MIN_PROTEIN_LAUK and prot > carb * 0.3

        # Berdasarkan keyword nama
        if any(l in words for l in _LAUK_KEYWORDS):
            if cat_lower in ('sayuran', 'buah', 'karbohidrat_pokok'):
                return False
            return prot >= MIN_PROTEIN_LAUK

        # Gorengan berbasis protein (tempe goreng, tahu goreng)
        if cat_lower == 'gorengan' and any(l in fname_lower for l in ['tempe', 'tahu', 'ayam', 'ikan', 'udang']):
            return prot >= MIN_PROTEIN_LAUK

        return False

    # ───── SAYUR / Pelengkap ─────────────────────────────────
    if role == 'Sayur':
        # Reject camilan/kue/olahan berat yang salah label
        if any(reject in fname_lower for reject in _REJECT_SAYUR_NAMES):
            return False

        # Sayur harus rendah kalori per 100g — batas realistis
        MAX_CAL_SAYUR = 200.0
        MAX_FAT_SAYUR = 15.0  # gram per 100g

        # Berdasarkan item_type
        if item_type == 'vegetable':
            # Hanya terima sayur yang bisa dimakan langsung (bukan kering/olahan)
            # Sayur segar biasanya < 200 kcal, tapi toleransi lebih untuk yang dimasak
            return cal <= 300.0 and fat <= 25.0

        if item_type == 'fruit':
            # Buah yang benar-benar buah: rendah lemak, kalori wajar
            return cal <= 200.0 and fat <= 10.0

        # Berdasarkan food_category
        if cat_lower == 'sayuran':
            return cal <= 300.0 and fat <= 25.0

        if cat_lower == 'buah':
            return cal <= 200.0 and fat <= 10.0

        # Berdasarkan keyword nama
        if any(s in fname_lower for s in _SAYUR_KEYWORDS):
            if cat_lower in ('lauk_hewani', 'lauk_nabati', 'karbohidrat_pokok'):
                return False
            return cal <= MAX_CAL_SAYUR and fat <= MAX_FAT_SAYUR

        return False

    return False


# ============================================================
# Internal Helpers
# ============================================================
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


def _row_pairing_group(row: pd.Series) -> str:
    return str(row.get('pairing_group', '')).lower().strip()


def _is_traditional_combo_compatible(karbo_row: pd.Series, lauk_row: pd.Series, sayur_row: pd.Series) -> bool:
    return (
        _row_pairing_group(karbo_row) in TRADITIONAL_STAPLE_GROUPS
        and _row_pairing_group(lauk_row) in TRADITIONAL_PROTEIN_GROUPS
        and _row_pairing_group(sayur_row) in TRADITIONAL_VEGETABLE_GROUPS
    )


def _is_sweet_combo_compatible(sweet_row: pd.Series, dairy_row: pd.Series, meal_type: str) -> bool:
    if meal_type != 'breakfast':
        return False
    return _row_pairing_group(sweet_row) in SWEET_STAPLE_GROUPS and _row_pairing_group(dairy_row) in DAIRY_PAIRING_GROUPS


def _best_traditional_combo(k_cands: list, l_cands: list, s_cands: list) -> Optional[Dict[str, tuple]]:
    best_combo, best_score = None, -np.inf
    for karbo in k_cands[:TOP_PAIRING_CANDIDATES]:
        for lauk in l_cands[:TOP_PAIRING_CANDIDATES]:
            if karbo[1] == lauk[1]:
                continue
            for sayur in s_cands[:TOP_PAIRING_CANDIDATES]:
                if sayur[1] in (karbo[1], lauk[1]):
                    continue
                if not _is_traditional_combo_compatible(karbo[3], lauk[3], sayur[3]):
                    continue
                score = (karbo[2] + lauk[2] + sayur[2]) / 3
                if score > best_score:
                    best_score = score
                    best_combo = {'Karbo': karbo, 'Lauk': lauk, 'Sayur': sayur, '_score': score}
    return best_combo


def _best_sweet_combo(sweet_cands: list, dairy_cands: list, meal_type: str) -> Optional[Dict[str, tuple]]:
    best_combo, best_score = None, -np.inf
    for sweet in sweet_cands[:TOP_PAIRING_CANDIDATES]:
        for dairy in dairy_cands[:TOP_PAIRING_CANDIDATES]:
            if sweet[1] == dairy[1]:
                continue
            if not _is_sweet_combo_compatible(sweet[3], dairy[3], meal_type):
                continue
            score = (sweet[2] + dairy[2]) / 2
            if score > best_score:
                best_score = score
                best_combo = {'Camilan/Roti': sweet, 'Pendamping': dairy, '_score': score}
    return best_combo


# ============================================================
# Combo Meal Recommendation
# ============================================================
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

    # Rasio makro per komponen
    target_k = [target_macros[0] * 0.45, target_macros[1] * 0.10, target_macros[2] * 0.10, target_macros[3] * 0.60]
    target_l = [target_macros[0] * 0.40, target_macros[1] * 0.75, target_macros[2] * 0.70, target_macros[3] * 0.10]
    target_s = [target_macros[0] * 0.15, target_macros[1] * 0.15, target_macros[2] * 0.20, target_macros[3] * 0.30]
    target_sweet = [target_macros[0] * 0.65, target_macros[1] * 0.25, target_macros[2] * 0.35, target_macros[3] * 0.75]
    target_dairy = [target_macros[0] * 0.35, target_macros[1] * 0.45, target_macros[2] * 0.35, target_macros[3] * 0.25]

    scores_full = predict_scores(target_macros)
    scores_k = predict_scores(target_k)
    scores_l = predict_scores(target_l)
    scores_s = predict_scores(target_s)
    scores_sweet = predict_scores(target_sweet)
    scores_dairy = predict_scores(target_dairy)

    full_cands, k_cands, l_cands, s_cands, sweet_cands, dairy_cands = [], [], [], [], [], []

    for i in range(num_foods):
        if has_allergy[i] or not suitable_mask[i]:
            continue

        row = filtered_food_df.iloc[i]
        fname = row.get('food_name', '')
        cal_100g = float(row.get('calories_100g', 0))
        if cal_100g <= 0:
            continue

        penalty = variety_penalty if fname in used_foods else 0

        # Strict categorization: setiap makanan hanya masuk SATU role.
        # Priority: complete menu, sweet breakfast pair, then traditional components.
        if is_valid_for_role(row, 'Complete'):
            full_cands.append((i, fname, scores_full[i] - penalty, row))
        elif is_valid_for_role(row, 'Sweet'):
            sweet_cands.append((i, fname, scores_sweet[i] - penalty, row))
        elif is_valid_for_role(row, 'Dairy'):
            dairy_cands.append((i, fname, scores_dairy[i] - penalty, row))
        elif is_valid_for_role(row, 'Karbo'):
            k_score = scores_k[i] + _karbo_priority_adjustment(row) - penalty - _karbo_variety_penalty(row, used_foods)
            k_cands.append((i, fname, k_score, row))
        elif is_valid_for_role(row, 'Lauk'):
            l_cands.append((i, fname, scores_l[i] - penalty, row))
        elif is_valid_for_role(row, 'Sayur'):
            s_cands.append((i, fname, scores_s[i] - penalty, row))

    full_cands.sort(key=lambda x: x[2], reverse=True)
    k_cands.sort(key=lambda x: x[2], reverse=True)
    l_cands.sort(key=lambda x: x[2], reverse=True)
    s_cands.sort(key=lambda x: x[2], reverse=True)
    sweet_cands.sort(key=lambda x: x[2], reverse=True)
    dairy_cands.sort(key=lambda x: x[2], reverse=True)

    result = {}
    traditional_combo = _best_traditional_combo(k_cands, l_cands, s_cands)
    sweet_combo = _best_sweet_combo(sweet_cands, dairy_cands, meal_type)
    best_full = full_cands[0] if full_cands else None

    options = []
    if best_full:
        options.append(('full', best_full[2], best_full))
    if traditional_combo:
        options.append(('traditional', traditional_combo['_score'], traditional_combo))
    if sweet_combo:
        # Sweet + dairy is intentionally a breakfast-only profile.
        options.append(('sweet', sweet_combo['_score'], sweet_combo))

    if not options:
        return result

    selected_type, _, selected = max(options, key=lambda x: x[1])

    def add_item(role: str, best: tuple, target_m: list, min_g: float, max_g: float) -> None:
        calories_100g = max(float(best[3].get('calories_100g', 1)), 1.0)
        grams = (target_m[0] / calories_100g) * 100
        grams = min(max(grams, min_g), max_g)
        result[role] = {
            'food_id': str(best[3].get('food_id', '')),
            'food_name': best[1],
            'image_url': str(best[3].get('image_url', '')) if best[3].get('image_url') is not None else '',
            'pairing_group': str(best[3].get('pairing_group', '')),
            'pairing_role': str(best[3].get('pairing_role', '')),
            'calories_100g': float(best[3].get('calories_100g', 0)),
            'grams': float(grams),
            'cal': float((best[3].get('calories_100g', 0) / 100) * grams),
            'prot': float((best[3].get('protein_100g', 0) / 100) * grams),
            'fat': float((best[3].get('fat_100g', 0) / 100) * grams),
            'carb': float((best[3].get('carbohydrate_100g', 0) / 100) * grams),
            'score': float(best[2]),
        }

    if selected_type == 'full':
        add_item('Makanan Utama', selected, target_macros, 120, 500)
    elif selected_type == 'sweet':
        add_item('Camilan/Roti', selected['Camilan/Roti'], target_sweet, 40, 180)
        add_item('Pendamping', selected['Pendamping'], target_dairy, 30, 300)
    else:
        add_item('Karbo', selected['Karbo'], target_k, 50, 250)
        add_item('Lauk', selected['Lauk'], target_l, 50, 250)
        add_item('Sayur', selected['Sayur'], target_s, 50, 250)

    return result


# ============================================================
# 7-Day Combo Meal Plan
# ============================================================
def generate_7day_combo_plan(
    daily_macros: list, user_allergy_list: list, meal_datasets: Optional[Dict[str, pd.DataFrame]],
    user_max: np.ndarray, variety_penalty: float = 0.20, start_date: Optional[str] = None, days: int = 7,
) -> List[Dict]:
    if loaded_model is None or days <= 0:
        return []

    base_date = datetime.strptime(start_date, '%Y-%m-%d') if start_date else datetime.today()
    used_foods, plan = set(), []

    for day_idx in range(days):
        current_date = base_date + timedelta(days=day_idx)
        day_label = DAY_NAMES[current_date.weekday()]
        day_used = set()

        for meal_key, (meal_label, ratio) in MEAL_RATIOS.items():
            meal_macros = [m * ratio for m in daily_macros]
            meal_df = meal_datasets.get(meal_key) if meal_datasets else None
            if meal_df is None:
                continue

            combo = get_combo_meal_recommendations(
                meal_type=meal_key, target_macros=meal_macros, user_allergy_list=user_allergy_list,
                filtered_food_df=meal_df, user_max=user_max, used_foods=used_foods | day_used,
                variety_penalty=variety_penalty,
            )

            recommendations = []
            for role in ['Makanan Utama', 'Camilan/Roti', 'Pendamping', 'Karbo', 'Lauk', 'Sayur']:
                if role not in combo:
                    continue
                item = combo[role]
                recommendations.append({
                    'food_id': item.get('food_id'),
                    'food_name': f"[{role}] {item['food_name']}",
                    'image_url': item.get('image_url'),
                    'pairing_group': item.get('pairing_group'),
                    'pairing_role': item.get('pairing_role'),
                    'calories_100g': item['calories_100g'],
                    'ideal_grams': item['grams'],
                    'ideal_calories': item['cal'],
                    'ideal_protein': item.get('prot', 0),
                    'ideal_fat': item.get('fat', 0),
                    'ideal_carb': item.get('carb', 0),
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
