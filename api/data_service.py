import pandas as pd
import numpy as np
import os

# ============================================================
# PATHS — Dataset v2 (kolom baru: is_recommendable_food, halal, dsb.)
# ============================================================
_BASE_DIR = os.path.dirname(__file__)
FOOD_DATA_PATH = os.path.join(_BASE_DIR, '..', 'Data', 'train_ready_dataset_v2.csv')
USER_DATA_PATH = os.path.join(_BASE_DIR, '..', 'Data', 'user_profile_features_schema.csv')


# ============================================================
# HELPER — Normalisasi kolom boolean (True/False/1/0/string)
# ============================================================
def _normalize_bool_col(series: pd.Series, default: bool = False) -> pd.Series:
    return (
        series.astype(str).str.strip().str.lower()
        .map({'true': True, 'false': False, '1': True, '0': False, '1.0': True, '0.0': False})
        .fillna(default)
        .astype(bool)
    )


def _normalize_int_col(series: pd.Series, default: int = 0) -> pd.Series:
    return (
        series.astype(str).str.strip().str.lower()
        .map({'true': 1, 'false': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0})
        .fillna(default)
        .astype(int)
    )


# ============================================================
# LOAD & CLEAN DATASET v2
# ============================================================
ALLERGY_COLS_FOOD = [
    'contains_gluten', 'contains_dairy', 'contains_nuts', 'contains_peanut',
    'contains_seafood', 'contains_egg', 'contains_soy', 'contains_celery',
]

MEAL_SUITABLE_COLS = ['suitable_breakfast', 'suitable_lunch', 'suitable_dinner']

ENFORCE_HALAL_FILTER_DEFAULT = True


def _load_and_clean_food_data(enforce_halal: bool = ENFORCE_HALAL_FILTER_DEFAULT) -> pd.DataFrame:
    """
    Load dataset v2 dan terapkan guardrail bawaan:
      1. Hanya makanan `is_recommendable_food == True`
      2. Buang `ingredient_only_flag` dan `raw_ingredient_flag`
      3. Opsional: filter halal (buang non-halal)
      4. Normalisasi kolom alergi & meal suitability
    """
    try:
        df = pd.read_csv(FOOD_DATA_PATH)
    except Exception as e:
        print(f"Error loading food CSV: {e}")
        return pd.DataFrame()

    before = len(df)

    # --- 1. is_recommendable_food ---
    if 'is_recommendable_food' in df.columns:
        df['is_recommendable_food'] = _normalize_bool_col(df['is_recommendable_food'], default=True)
        df = df[df['is_recommendable_food']]

    # --- 2. ingredient_only / raw_ingredient ---
    for flag_col in ['ingredient_only_flag', 'raw_ingredient_flag']:
        if flag_col in df.columns:
            df[flag_col] = _normalize_bool_col(df[flag_col], default=False)
            df = df[~df[flag_col]]

    # --- 3. Halal guardrail ---
    if enforce_halal:
        if 'contains_non_halal_ingredient' in df.columns:
            df['contains_non_halal_ingredient'] = _normalize_bool_col(df['contains_non_halal_ingredient'], default=False)
            df = df[~df['contains_non_halal_ingredient']]
        if 'is_halal_candidate' in df.columns:
            df['is_halal_candidate'] = _normalize_bool_col(df['is_halal_candidate'], default=True)
            df = df[df['is_halal_candidate']]
        if 'halal_status' in df.columns:
            df = df[df['halal_status'].fillna('halal_candidate').eq('halal_candidate')]

    # --- 4. Normalisasi kolom alergi → float ---
    for col in ALLERGY_COLS_FOOD:
        if col in df.columns:
            df[col] = _normalize_int_col(df[col], default=0).astype(np.float32)
        else:
            df[col] = np.float32(0)

    # --- 5. Normalisasi meal suitability → int ---
    for col in MEAL_SUITABLE_COLS:
        if col in df.columns:
            df[col] = _normalize_int_col(df[col], default=1)
        else:
            df[col] = 1  # default cocok semua waktu

    # --- 6. Pastikan kolom makro numerik ---
    for col in ['calories_100g', 'protein_100g', 'fat_100g', 'carbohydrate_100g']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df = df.reset_index(drop=True)
    removed = before - len(df)
    print(f"Food data loaded: {len(df)} rows (removed {removed} by v2 guardrails) from {FOOD_DATA_PATH}")
    return df


# ── Init ─────────────────────────────────────────────────────
food_df = _load_and_clean_food_data(enforce_halal=ENFORCE_HALAL_FILTER_DEFAULT)

try:
    user_df = pd.read_csv(USER_DATA_PATH)
except Exception as e:
    print(f"Error loading user CSV: {e}")
    user_df = pd.DataFrame()


# ============================================================
# PUBLIC API
# ============================================================
def get_food_data() -> pd.DataFrame:
    return food_df


def filter_foods_by_preferences(
    categories: list,
    ingredients: list,
    target_meal: str,
    dataset: pd.DataFrame,
    negative_keywords: list = None,
    halal_only: bool = False,
) -> pd.DataFrame:
    """
    Filter dataset berdasarkan kategori, bahan, waktu makan, keyword negatif,
    dan opsional halal dinamis (untuk kasus halal_only belum diterapkan saat load).
    """
    filtered_df = dataset.copy()

    # 0. Negative Filtering
    if negative_keywords:
        neg_lower = [n.lower() for n in negative_keywords]
        for neg in neg_lower:
            filtered_df = filtered_df[
                ~(
                    filtered_df['food_name'].fillna('').str.lower().str.contains(neg, regex=False) |
                    filtered_df['food_category'].fillna('').str.lower().str.contains(neg, regex=False) |
                    filtered_df['main_ingredient'].fillna('').str.lower().str.contains(neg, regex=False)
                )
            ]

    # 1. Filter by Food Category
    if categories:
        cat_lower = [c.lower() for c in categories]
        filtered_df = filtered_df[filtered_df['food_category'].fillna('').str.lower().isin(cat_lower)]

    # 2. Filter by Main Ingredient
    if ingredients:
        ing_lower = [i.lower() for i in ingredients]
        filtered_df = filtered_df[filtered_df['main_ingredient'].fillna('').str.lower().isin(ing_lower)]

    # 3. Filter by Meal Time — gunakan kolom boolean v2
    if target_meal:
        target_meal_lower = target_meal.lower()
        meal_filtered = None

        if "sarapan" in target_meal_lower:
            if 'suitable_breakfast' in filtered_df.columns:
                meal_filtered = filtered_df[filtered_df['suitable_breakfast'] == 1]
        elif "siang" in target_meal_lower:
            if 'suitable_lunch' in filtered_df.columns:
                meal_filtered = filtered_df[filtered_df['suitable_lunch'] == 1]
        elif "malam" in target_meal_lower:
            if 'suitable_dinner' in filtered_df.columns:
                meal_filtered = filtered_df[filtered_df['suitable_dinner'] == 1]

        if meal_filtered is not None and not meal_filtered.empty:
            filtered_df = meal_filtered

    # 4. Dynamic halal filter (jika belum diterapkan saat load)
    if halal_only:
        if 'contains_non_halal_ingredient' in filtered_df.columns:
            filtered_df = filtered_df[~_normalize_bool_col(filtered_df['contains_non_halal_ingredient'], default=False)]
        if 'is_halal_candidate' in filtered_df.columns:
            filtered_df = filtered_df[_normalize_bool_col(filtered_df['is_halal_candidate'], default=True)]

    # 5. Fallback jika filter terlalu ketat
    if filtered_df.empty:
        print(f"Warning: No foods found for prefs (cat={categories}, ing={ingredients}, meal={target_meal}). Falling back to full dataset.")
        return dataset

    return filtered_df


# ============================================================
# Macro normalization factors (berdasarkan user profile)
# ============================================================
user_macro_cols = ['target_calorie', 'protein_target_g', 'fat_target_g', 'carb_target_g']
if not user_df.empty:
    user_max = np.max(user_df[user_macro_cols].fillna(0).values, axis=0) + 1e-9
else:
    user_max = np.array([3000.0, 200.0, 100.0, 400.0])  # Fallback sensible max


def get_user_max():
    return user_max
