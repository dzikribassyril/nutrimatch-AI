import pandas as pd
import numpy as np

# Load Data Once
FOOD_DATA_PATH = '../Data/train_ready_dataset.csv'
USER_DATA_PATH = '../Data/user_profile_features_schema.csv'

try:
    food_df = pd.read_csv(FOOD_DATA_PATH)
    user_df = pd.read_csv(USER_DATA_PATH)
except Exception as e:
    print(f"Error loading CSV files: {e}")
    food_df = pd.DataFrame()
    user_df = pd.DataFrame()

def get_food_data():
    return food_df

def filter_foods_by_preferences(categories: list, ingredients: list, target_meal: str, dataset: pd.DataFrame, negative_keywords: list = None) -> pd.DataFrame:
    """
    Filters food dataset based on precise category, main ingredient, and meal time 
    provided by the Data Science team. Also applies negative filtering.
    """
    filtered_df = dataset.copy()
    
    # 0. Negative Filtering
    if negative_keywords:
        neg_lower = [n.lower() for n in negative_keywords]
        for neg in neg_lower:
            # Drop rows where food_name, food_category, or main_ingredient contains the negative word
            filtered_df = filtered_df[
                ~(
                    filtered_df['food_name'].fillna('').str.lower().str.contains(neg) |
                    filtered_df['food_category'].fillna('').str.lower().str.contains(neg) |
                    filtered_df['main_ingredient'].fillna('').str.lower().str.contains(neg)
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
        
    # 3. Filter by Meal Time using DS Boolean Flags
    if target_meal:
        target_meal_lower = target_meal.lower()
        # Jika DF menjadi kosong setelah filter meal, kita lewati agar tidak error (fallback)
        meal_filtered = None
        def _normalize_suitable(series: pd.Series) -> pd.Series:
            return (
                series.astype(str)
                .str.lower()
                .map({'true': 1, 'false': 0, '1': 1, '0': 0})
                .fillna(0)
                .astype(int)
            )

        if "sarapan" in target_meal_lower:
            if 'suitable_breakfast' in filtered_df.columns:
                meal_filtered = filtered_df[_normalize_suitable(filtered_df['suitable_breakfast']) == 1]
        elif "siang" in target_meal_lower:
            if 'suitable_lunch' in filtered_df.columns:
                meal_filtered = filtered_df[_normalize_suitable(filtered_df['suitable_lunch']) == 1]
        elif "malam" in target_meal_lower:
            if 'suitable_dinner' in filtered_df.columns:
                meal_filtered = filtered_df[_normalize_suitable(filtered_df['suitable_dinner']) == 1]
            
        if meal_filtered is not None and not meal_filtered.empty:
            filtered_df = meal_filtered
            
    # 4. Fallback jika filter terlalu ketat hingga kosong
    if filtered_df.empty:
        print(f"Warning: No foods found for prefs (cat={categories}, ing={ingredients}, meal={target_meal}). Falling back to full dataset.")
        return dataset
        
    return filtered_df

# Macro normalization factors based on the training data limits
# We approximate these from the dataset
user_macro_cols = ['target_calorie', 'protein_target_g', 'fat_target_g', 'carb_target_g']
if not user_df.empty:
    user_max = np.max(user_df[user_macro_cols].fillna(0).values, axis=0) + 1e-9
else:
    user_max = np.array([3000.0, 200.0, 100.0, 400.0]) # Fallback sensible max values

def get_user_max():
    return user_max
